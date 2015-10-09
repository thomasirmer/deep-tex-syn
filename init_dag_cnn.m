function [net, objectiveString] = init_dag_cnn(im, nets, opts)
% Sanity check
layerNames = cellfun(@(x) x.name, nets.layers, 'UniformOutput', false);
[membership, id] = ismember(opts.textureLayer, layerNames);
assert(all(membership));
nets.layers = nets.layers(1:max(id)); %Clip nets to the maxLayer

net = dagnn.DagNN();
net = net.fromSimpleNN(nets, 'CanonicalNames', true);
for i = 1:length(opts.textureLayer), 
    % Add bilinear layer with inputs and target outputs
    layerOutput = net.layers(net.getLayerIndex(opts.textureLayer{i})).outputs;
    assert(length(layerOutput) == 1);
    input = layerOutput{1};
    layerName = sprintf('b%s',opts.textureLayer{i});
    output = sprintf('tex%i', i);
    net.addLayer(layerName, dagnn.BilinearPooling(), {input}, output);
    
    % Add a loss layer matching the input to the output
    input2 = sprintf('target%i', i); 
    layerName = sprintf('loss%i', i);
    output2 = sprintf('objective%i', i);
    net.addLayer(layerName, dagnn.Loss('loss', 'l2'), ...
        {output,input2},output2) ;
end
    
% Compute targets to match
im = imresize(im, nets.normalization.imageSize([2 1]));
im_ = single(im);
im_ = im_ - nets.normalization.averageImage;
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
for i = 1:length(opts.textureLayer);
    objectiveString = {objectiveString{:}, ...
                    sprintf('objective%i', i), opts.textureLayerWeights(i)};
end
net.conserveMemory = false;
