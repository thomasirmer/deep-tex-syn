function info = invert_attribute()
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

classNames = tmp.classes;
for i = 1:length(classNames), 
    fprintf('class %i/%i: %s\n', i, length(classNames), classNames{i});
    outFile = fullfile(opts.expDir, sprintf('%s.png', classNames{i}));
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
                        'attributeTarget', classNames{i},...
                        'attributeDir', opts.attributeDir, ...
                        'useGPU', true);
          
    imwrite(gather(imsyn), outFile);
    if false,
        figure(1); clf;
        imshow(imsyn); axis image off;
        title(sprintf('inverse: %s', classNames{i}));
    end
end
