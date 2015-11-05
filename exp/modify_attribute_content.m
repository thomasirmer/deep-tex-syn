function modify_attribute_content(net, varargin)
% Set paths
opts.prefix = 'v2';
opts.attribute = 'dtd';
opts = vl_argparse(opts, varargin);
opts.expDir = fullfile('data', opts.prefix, opts.attribute);
opts.attributeDir = fullfile('data', 'models', opts.attribute);

% Create directories
mkdir(opts.expDir);
tmp = load(fullfile(opts.attributeDir, 'relu2_2.mat'));

classNames = tmp.classes;

imageName = 'jolie.jpg';
im = imread(fullfile('data', 'textures', imageName));
im = imresize(im, [224 224]);

for i = 1:length(classNames), 
    fprintf('class %i/%i: %s\n', i, length(classNames), classNames{i});
    outFile = fullfile(opts.expDir, sprintf('%s-%s.png', ...
                                            imageName(1:end-4), classNames{i}));
    if exist(outFile, 'file');
        continue;
    end
    rand('seed', 0);
    res = texture_syn(im, net, ...
                      'imageSize', [224 224 3], ...
                      'TVbeta', 2, ...
                      'lambdaTV', 1e-6, ...
                      'beta', 2, ...
                      'lambdaLb', 0, ...
                      'contentLayer', {'relu4_2'}, ...
                      'contentLayerWeights', [1], ...
                      'textureLayer', {}, ...
                      'textureLayerWeights', [1], ...
                      'attributeLayer', {'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'}, ...
                      'attributeLayerWeights', [1 1 1 1]*1000, ...
                      'attributeTarget', {classNames{i}},...
                      'attributeDir', opts.attributeDir, ...
                      'textureInit', 'rand',...
                      'maxIter', 100, ...
                      'useGPU', false);
    
    imwrite(gather(res.imsyn), outFile);
    if true,
        figure(1); clf;
        imshow(res.imsyn); axis image off;
        title(sprintf('inverse: %s', classNames{i}));
    end
end