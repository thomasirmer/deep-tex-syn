function [net, objectiveString] = init_dag_cnn(im, nets, opts)
% Sanity check
layerNames = cellfun(@(x) x.name, nets.layers, 'UniformOutput', false);
[membership, id] = ismember({opts.contentLayer{:} opts.textureLayer{:} opts.attributeLayer{:}}, layerNames);
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
    output = sprintf('style_%i', i);
    net.addLayer(layerName, BilinearPooling('normalizeGradients', true), ...
                 {input}, output);

    % Add a loss layer matching the input to the output
    input = output;
    input2 = sprintf('target_style_%i', i);
    layerName = sprintf('loss_style_%i', i);
    output = sprintf('obj_style_%i', i);
    net.addLayer(layerName, L2Loss('loss', 'l2'), {input,input2},output) ;
end

% For each content layer in opts.contentLayer
for i = 1:length(opts.contentLayer),
   % Add loss layer
   input = net.layers(net.getLayerIndex(opts.contentLayer{i})).outputs{1};
   input2 = sprintf('target_cont_%i', i);
   layerName = sprintf('loss_cont_%i', i);
   output = sprintf('obj_cont_%i', i);
   net.addLayer(layerName, L2Loss('loss', 'l2'), {input, input2}, output);
end

% For each attribute layer in opts.attributeLayer
for i = 1:length(opts.attributeLayer),
    % Add bilinearpool layer (unless it already exists)
    [alreadyComputed, memId] = ismember(opts.attributeLayer{i}, opts.textureLayer);
    if alreadyComputed
        output = sprintf('style_%i', memId);
    else
        layerOutput = net.layers(net.getLayerIndex(opts.attributeLayer{i})).outputs;
        input = layerOutput{1};
        layerName = sprintf('ba%s',opts.attributeLayer{i});
        output = sprintf('stylea_%i', i);
        net.addLayer(layerName, ...
                     BilinearPooling('normalizeGradients', false), {input}, output);
    end
    
    % Square-root layer
    layerName = sprintf('sqrt%s', opts.attributeLayer{i});
    input = output;
    output = sprintf('sqrta%i', i);
    net.addLayer(layerName, SquareRoot(), {input}, output);
    
    % L2 normalization layer
    layerName = sprintf('l2%s', opts.attributeLayer{i});
    input = output;
    output = sprintf('l2a%i', i);
    net.addLayer(layerName, L2Norm(), {input}, output);
    
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

    
    % Loss layer (Softmax loss) for each attribute target
    input = output;
    for a = 1:length(opts.attributeTarget), 
        inputattr = sprintf('target_attr_%i', a);
        layerName = sprintf('loss_attr_%i_%i', i, a);
        output = sprintf('obj_attr_%i_%i',i,a);
        net.addLayer(layerName, dagnn.Loss('loss', 'softmaxlog'), ...
                     {input,inputattr}, output) ;
    end
end

% Compute targets to match the texture outputs
meanRGB = mean(mean(nets.normalization.averageImage));
im_ = single(im);
im_ = bsxfun(@minus, im_, meanRGB);
for i = 1:length(opts.textureLayer), 
    texInd = net.getVarIndex(sprintf('style_%i',i));
    net.vars(texInd).precious = true;
end
for i = 1:length(opts.contentLayer), 
    input = net.layers(net.getLayerIndex(opts.contentLayer{i})).outputs{1};
    styleInd = net.getVarIndex(input);
    net.vars(styleInd).precious = true;
end

net.conserveMemory = false;
if opts.useGPU
  im_ = gpuArray(im_);
  net.move('gpu');
end
net.eval({'input', im_});

% Set the target values for style
for i = 1:length(opts.textureLayer), 
    texInd = net.getVarIndex(sprintf('style_%i', i));
    targetInd = net.getVarIndex(sprintf('target_style_%i', i));
    objectiveInd = net.getVarIndex(sprintf('obj_style_%i',i));
    net.vars(targetInd).value = net.vars(texInd).value;
    net.vars(targetInd).precious = true;
    net.vars(texInd).value = [];
    net.vars(objectiveInd).precious = true;
    net.vars(texInd).precious = false;
end

% Set target values for content
for i = 1:length(opts.contentLayer), 
    input = net.layers(net.getLayerIndex(opts.contentLayer{i})).outputs{1};
    styleInd = net.getVarIndex(input);
    targetInd = net.getVarIndex(sprintf('target_cont_%i', i));
    objectiveInd = net.getVarIndex(sprintf('obj_cont_%i',i));
    net.vars(targetInd).value = net.vars(styleInd).value;
    net.vars(targetInd).precious = true;
    net.vars(styleInd).value = [];
    net.vars(objectiveInd).precious = true;
    net.vars(styleInd).precious = false;
end

% Set target attribute values
for a = 1:length(opts.attributeTarget)
    [~, classId] = ismember(opts.attributeTarget{a}, tmp.classes);
    varId = net.getVarIndex(sprintf('target_attr_%i',a));
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
                       sprintf('obj_style_%i', i), opts.textureLayerWeights(i)};
end
for i = 1:length(opts.contentLayer),
    objectiveString = {objectiveString{:}, ...
                       sprintf('obj_cont_%i', i), opts.contentLayerWeights(i)};
end
for i = 1:length(opts.attributeLayer)
    for a = 1:length(opts.attributeTarget)
    objectiveString = {objectiveString{:}, ...
                       sprintf('obj_attr_%i_%i', i,a), ...
                       opts.attributeLayerWeights(i)* ...
                       opts.attributeTargetWeights(a)};
    end
end    
net.conserveMemory = false;