function info = blend_attribute(varargin)
% Load network
if ~exist('net', 'var')
    net = load('data/models/imagenet-vgg-verydeep-16.mat');
end

% Set paths
opts.prefix = 'v1';
opts.attribute = 'dtd';
opts = vl_argparse(opts, varargin);
opts.expDir = fullfile('data', opts.prefix, opts.attribute);
opts.attributeDir = fullfile('data', 'models', opts.attribute);

% Create directories
mkdir(opts.expDir);
tmp = load(fullfile(opts.attributeDir, 'relu2_2.mat'));

attributeTarget = {{'swirly', 'paisley'}};
attributeTargetWeights = {[0.5 1]};

for i = 1:length(attributeTarget), 
    fprintf('class %i/%i: %s\n', i, length(attributeTarget), makeFilename(attributeTarget{i}));
    outFile = fullfile(opts.expDir, sprintf('%s.png', makeFilename(attributeTarget{i})));
    if exist(outFile, 'file');
        continue;
    end
    res = texture_syn([], net, ...
                      'imageSize', [224 224 3], ...
                      'TVbeta', 2, ...
                      'lambdaTV', 1e-6, ...
                      'beta', 2, ...
                      'lambdaLb', 0, ...
                      'contentLayer', {}, ...
                      'contentLayerWeights', [], ...
                      'textureLayer', {}, ...
                      'textureLayerWeights', [], ...
                      'attributeLayer', {'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'}, ...
                      'attributeLayerWeights', [1 1 1 1]*100, ...
                      'attributeTarget', attributeTarget{i},...
                      'attributeTargetWeights', attributeTargetWeights{i}, ...
                      'attributeDir', opts.attributeDir, ...
                      'maxIter', 100, ...
                      'useGPU', false);
          
    imwrite(gather(res.imsyn), outFile);
    if false,
        figure(1); clf;
        imshow(imsyn); axis image off;
        title(sprintf('inverse: %s', classNames{i}));
    end
    info.res{i} = res;
end
info.attributeTarget = attributeTarget;
info.attributeTargetWeights = attributeTargetWeights;

function filename=makeFilename(attr)
filename = attr{1};
for i = 2:length(attr)
    filename = sprintf('%s-%s', filename,attr{i});
end
