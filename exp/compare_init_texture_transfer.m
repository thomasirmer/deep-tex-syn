function info = compare_init_texture_transfer()
% Experiment comparing different initialization techniques

opts.prefix = 'v2-text-trans-2000';
opts.attribute = 'dtd';
opts.expDir = fullfile('data', opts.prefix, opts.attribute);
opts.attributeDir = fullfile('data', 'models', opts.attribute);
opts.imageName = 'jolie.jpg';

% Create directories
mkdir(opts.expDir);


% Load network
if ~exist('net', 'var')
    net = load('data/models/imagenet-vgg-verydeep-16.mat');
end


tmp = load(fullfile(opts.attributeDir, 'relu2_2.mat'));
classNames = tmp.classes;

classNames = {classNames{find(strcmp('marbled', classNames))}};

imPath = fullfile('data', 'textures', opts.imageName);
im = imread(imPath);
[~, imName, ~] = fileparts(imPath);
im = imresize(im, [224 224]);


% Run texture synthesis
commonArgs = {'imageSize', [224 224 3], ...
              'TVbeta', 2, ...
              'lambdaTV', 1e-6, ...
              'beta', 2, ...
              'lambdaLb', 0, ...
              'contentLayer', {'relu4_2'}, ...
              'contentLayerWeights', [5e-8], ...
              'textureLayer', {}, ...
              'textureLayerWeights', [1], ...
              'attributeLayer', {'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'}, ...
              'attributeLayerWeights', [1 1 1 1]*2000, ...
              'attributeTarget', {}, ...
              'attributeDir', opts.attributeDir, ...
              'maxIter', 1000, ...
              'useGPU', true};

numRuns = 5;

for j=1:length(classNames)
    
    if(exist(fullfile(opts.expDir, 'res', [imName, '-', classNames{j}, '.mat'])))
        continue;
    end

    clear info
    for i = 1:numRuns,
        fprintf('run %i/%i: random\n', i, numRuns);
        info.resRand{i} = texture_syn(im, net, ...
            {commonArgs{:}, 'textureInit', 'rand', 'attributeTarget', {classNames{j}}});
        fprintf('run %i/%i: quilt\n', i, numRuns);
        info.resQuilt{i} = texture_syn(im, net, ...
            {commonArgs{:}, 'textureInit', 'transfer-quilt', 'attributeTarget', {classNames{j}}});
    end
    info.commonArgs = commonArgs;
    
    % Do some plotting
    figure(999); clf;
    for i = 1:numRuns,
        semilogy(info.resRand{i}.info.trace.fval, 'r-'); hold on;
        semilogy(info.resQuilt{i}.info.trace.fval, 'b-');
    end
    legend('rand', 'quilt');
    xlabel('iter');
    ylabel('objective');
    title('compare init');
    
    
    if ~isempty(which('export_fig'))
        if ~exist(fullfile(opts.expDir, 'plot'), 'dir'), mkdir(fullfile(opts.expDir, 'plot')); end
        figure(999)
        export_fig('transparent', 'pdf', fullfile(opts.expDir, 'plot', [imName, '-', classNames{j}]));
    end
    
    figure(1000);
    for i = 1:numRuns,
        vl_tightsubplot(3, numRuns, i, 'margin', 0.01); imagesc(info.resRand{i}.imsyn); axis image off;
        vl_tightsubplot(3, numRuns, numRuns+i, 'margin', 0.01); imagesc(info.resQuilt{i}.init); axis image off;
        vl_tightsubplot(3, numRuns, 2*numRuns+i, 'margin', 0.01); imagesc(info.resQuilt{i}.imsyn); axis image off;
    end
     
    if ~isempty(which('export_fig'))
        if ~exist(fullfile(opts.expDir, 'syn'), 'dir'), mkdir(fullfile(opts.expDir, 'syn')); end
        figure(1000)
        export_fig('transparent', 'jpg', fullfile(opts.expDir, 'syn', [imName, '-', classNames{j}]));
    end
    
    if ~exist(fullfile(opts.expDir, 'res'), 'dir'), mkdir(fullfile(opts.expDir, 'res')); end
    save(fullfile(opts.expDir, 'res', [imName, '-', classNames{j}]), 'info', '-v7.3');
    
   close all 
end