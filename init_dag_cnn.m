function [net, objectiveString] = init_dag_cnn(im, nets, opts)
% Sanity check
layerNames = cellfun(@(x) x.name, nets.layers, 'UniformOutput', false);
[membership, id] = ismember({opts.textureLayer{:} opts.attributeLayer{:}}, layerNames);
assert(all(membership));
nets.layers = nets.layers(1:max(id)); %Clip nets to the maxLayer

% Convert simplenn to dagnn
net = dagnn.DagNN();
net = net.fromSimpleNN(nets, 'CanonicalNames', true);

% For each texture layer in opts.textureLayer, add a bilinear layer
% with l2 loss layers for matching the outputs
for i = 1:length(opts.textureLayer),
    % Add bilinearpool layer
    layerOutput = net.layers(net.getLayerIndex(opts.textureLayer{i})).outputs;
    assert(length(layerOutput) == 1);
    input = layerOutput{1};
    layerName = sprintf('b%s', opts.textureLayer{i});
    output = sprintf('b%i', i);
    net.addLayer(layerName, dagnn.BilinearPooling(), {input}, output);

    % Square-root layer
    layerName = sprintf('sqrt%s', opts.textureLayer{i});
    input = output;
    output = sprintf('sqrt%i', i);
    net.addLayer(layerName, dagnn.SquareRoot(), {input}, output);
    
    % L2 normalization layer
    layerName = sprintf('l2%s', opts.textureLayer{i});
    input = output;
    output = sprintf('tex%i', i);
    net.addLayer(layerName, dagnn.L2Norm(), {input}, output);

    % Add a loss layer matching the input to the output
    input = output;
    input2 = sprintf('target%i', i);
    layerName = sprintf('loss%i', i);
    output = sprintf('objective%i', i);
    net.addLayer(layerName, dagnn.Loss('loss', 'l2'), ...
        {input,input2},output) ;
end

% For each attribute layer in opts.attributeLayer, add a bilinear
% layer with sqrt, l2 normalization, and logistic loss layers
for i = 1:length(opts.attributeLayer),
    layerOutput = net.layers(net.getLayerIndex(opts.attributeLayer{i})).outputs;
    assert(length(layerOutput) == 1);
    input = layerOutput{1};
    layerName = sprintf('b%s',opts.attributeLayer{i});
    output = sprintf('tex%i', i);
    
    % Add bilinearpool layer (unless it already exists)
    alreadyComputed = ismember(opts.attributeLayer, ...
                               opts.textureLayer);
    if ~alreadyComputed
        net.addLayer(layerName, dagnn.BilinearPooling(), {input}, ...
                     output);
    end
    
    % Square-root layer
    layerName = sprintf('sqrt%s', opts.attributeLayer{i});
    input = output;
    output = sprintf('sqrttex%i', i);
    net.addLayer(layerName, dagnn.SquareRoot(), {input}, output);
    
    % L2 normalization layer
    layerName = sprintf('l2%s', opts.attributeLayer{i});
    input = output;
    output = sprintf('l2tex%i', i);
    net.addLayer(layerName, dagnn.L2Norm(), {input}, output);
    
    % Convolutional layer
    layerName = sprintf('attr%i', opts.attributeLayer{i});
    input = output;
    output = sprintf('score%i', i);

    % TODO: add parameters based on the name of the file and set
    % these when creating layers
    param(1).name = sprintf('wts%i', opts.attributeLayer{i});
    %param(1).value = ...;
    param(2).name = sprintf('bias%i', opts.attributeLayer{i});
    %param(2).value = ...;
    net.addLayer(layerName, dagnn.Conv(), {input}, output, param.name);

    % Softmax layer
    layerName = sprintf('prob%i', opts.attributeLayer{i});
    input = output; 
    output = sprintf('prob%i', i);
    net.addLayer(layerName, dagnn.SoftMax(), {input}, output);
    
    % Loss layer (Softmax loss)
    input = output;
    inputattr = sprintf('targetattr');
    layerName = sprintf('loss-attr%i', i);
    output = sprintf('objective-attr%i',i);
    net.addLayer(layerName, dagnn.Loss('loss', 'softmax'), ...
        {input,inputattr},output) ;
end

% Compute targets to match the texture outputs
meanRGB = mean(mean(nets.normalization.averageImage));
im_ = single(im);
im_ = bsxfun(@minus, im_, meanRGB);
for i = 1:length(opts.textureLayer), 
    texInd = net.getVarIndex(sprintf('tex%i',i));
    net.vars(texInd).precious = true;
end
net.conserveMemory = false;
if opts.useGPU
  im_ = gpuArray(im_);
  net.move('gpu');
end
net.eval({'input', im_});

% Set the target values
for i = 1:length(opts.textureLayer), 
    texInd = net.getVarIndex(sprintf('tex%i', i));
    targetInd = net.getVarIndex(sprintf('target%i', i));
    objectiveInd = net.getVarIndex(sprintf('objective%i',i));
    net.vars(targetInd).value = net.vars(texInd).value;
    net.vars(targetInd).precious = true;
    net.vars(texInd).value = [];
    net.vars(objectiveInd).precious = true;
    net.vars(texInd).precious = false;
end

% Set a string of weighted objectives;
objectiveString = {};
for i = 1:length(opts.textureLayer),
    objectiveString = {objectiveString{:}, ...
                    sprintf('objective%i', i), opts.textureLayerWeights(i)};
end
for i = 1:length(opts.attributeLayer)
    objectiveString = {objectiveString{:}, ...
                    sprintf('objective-attr%i', i), opts.attributeLayerWeights(i)};
end    
net.conserveMemory = false;