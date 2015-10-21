function [net, objectiveString] = init_dag_cnn(im, nets, opts)
% Sanity check
layerNames = cellfun(@(x) x.name, nets.layers, 'UniformOutput', false);
[membership, id] = ismember({opts.textureLayer{:} opts.attributeLayer{:}}, layerNames);
assert(all(membership));
nets.layers = nets.layers(1:max(id)); %Clip nets to the maxLayer

% Convert simplenn to dagnn
net = dagnn.DagNN();
net = net.fromSimpleNN(nets, 'CanonicalNames', true);

% For each texture layer in opts.textureLayer
for i = 1:length(opts.textureLayer),
    % Add bilinearpool layer
    layerOutput = net.layers(net.getLayerIndex(opts.textureLayer{i})).outputs;
    input = layerOutput{1};
    layerName = sprintf('b%s', opts.textureLayer{i});
    output = sprintf('tex%i', i);
    net.addLayer(layerName, dagnn.BilinearPooling('normalizeGradients', true), ...
                 {input}, output);

    % Add a loss layer matching the input to the output
    input = output;
    input2 = sprintf('target%i', i);
    layerName = sprintf('loss%i', i);
    output = sprintf('objective%i', i);
    net.addLayer(layerName, dagnn.Loss('loss', 'l2'), ...
        {input,input2},output) ;
end

% For each attribute layer in opts.attributeLayer
for i = 1:length(opts.attributeLayer),
    % Add bilinearpool layer (unless it already exists)
    [alreadyComputed, memId] = ismember(opts.attributeLayer{i}, opts.textureLayer);
    if alreadyComputed
        output = sprintf('tex%i', memId);
    else
        layerOutput = net.layers(net.getLayerIndex(opts.attributeLayer{i})).outputs;
        input = layerOutput{1};
        layerName = sprintf('ba%s',opts.attributeLayer{i});
        output = sprintf('texa%i', i);
        net.addLayer(layerName, ...
                     dagnn.BilinearPooling('normalizeGradients', false), {input}, output);
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
    layerName = sprintf('attr%s', opts.attributeLayer{i});
    input = output;
    output = sprintf('score%i', i);
    modelfile = fullfile(opts.attributeDir, opts.attributeLayer{i});
    tmp = load(modelfile);
    param(1).name = sprintf('convattr_%if', i);
    param(1).value = reshape(tmp.w, [1 1 size(tmp.w,1) size(tmp.w,2)]);
    param(2).name = sprintf('convattr_%ib', i);
    param(2).value = tmp.b;
    net.addLayer(layerName, dagnn.Conv(), {input}, output, {param(1).name param(2).name});
    for f = 1:2,
	varId = net.getParamIndex(param(f).name);
        if strcmp(net.device, 'gpu')
	  net.params(varId).value = gpuArray(param(f).value);
        else
	  net.params(varId).value = param(f).value;
        end
    end

    % Softmax layer
    layerName = sprintf('prob%s', opts.attributeLayer{i});
    input = output; 
    output = sprintf('proba%i', i);
    net.addLayer(layerName, dagnn.SoftMax(), {input}, output);
    
    % Loss layer (Softmax loss)
    input = output;
    inputattr = sprintf('targetattr');
    layerName = sprintf('lossattr%i', i);
    output = sprintf('objectiveattr%i',i);
    net.addLayer(layerName, dagnn.Loss('loss', 'softmaxlog'), ...
                                                {input,inputattr}, output) ;
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

% Set target attribute values
if length(opts.attributeLayer) > 0
    classes = tmp.classes;
    [~, classId] = ismember(opts.attributeTarget, tmp.classes);
    varId = net.getVarIndex('targetattr');
    targetLabel = zeros([1 1 1 1],'single');
    targetLabel = classId;
    if opts.useGPU
        net.vars(varId).value = gpuArray(targetLabel);
    else
        net.vars(varId).value = targetLabel;
    end
    net.vars(varId).precious = true;
end


% Set a string of weighted objectives;
objectiveString = {};
for i = 1:length(opts.textureLayer),
    objectiveString = {objectiveString{:}, ...
                    sprintf('objective%i', i), opts.textureLayerWeights(i)};
end
for i = 1:length(opts.attributeLayer)
    objectiveString = {objectiveString{:}, ...
                    sprintf('objectiveattr%i', i), opts.attributeLayerWeights(i)};
end    
net.conserveMemory = false;