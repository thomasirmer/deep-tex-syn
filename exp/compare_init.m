function info = compare_init()
% Experiment comparing different initialization techniques
setup;

% Load network
if ~exist('net', 'var')
    net = load('data/models/imagenet-vgg-verydeep-16.mat');
end

im = imread('data/textures/onion.png');
im = imresize(im, [224 224]);


% Run texture synthesis
commonArgs = {'imageSize', [224 224 3], ...
              'TVbeta', 2, ...
              'lambdaTV', 1e-6, ...
              'beta', 2, ...
              'lambdaLb', 0, ...
              'contentLayer', {}, ...
              'contentLayerWeights', [], ...
              'textureLayer', {'relu1_1','relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'}, ...
              'textureLayerWeights', [1 1 1 1 1], ...
              'attributeLayer', {}, ...
              'attributeLayerWeights', [], ...
              'maxIter', 250, ...
              'useGPU', true};

numRuns = 5;

for i = 1:numRuns,
    fprintf('run %i/%i: random\n', i, numRuns);
    info.resRand{i} = texture_syn(im, net, ...
                                  {commonArgs{:}, 'textureInit', 'rand'});
    fprintf('run %i/%i: quilt\n', i, numRuns);
    info.resQuilt{i} = texture_syn(im, net, ...
                                   {commonArgs{:}, 'textureInit', 'quilt'});
end
info.commonArgs = commonArgs;

% Do some plotting
figure; clf;
for i = 1:numRuns,
    semilogy(info.resRand{i}.info.trace.fval, 'r-'); hold on;
    semilogy(info.resQuilt{i}.info.trace.fval, 'b-');
end
legend('rand', 'quilt');
xlabel('iter');
ylabel('objective');
title('compare init');


figure;
for i = 1:numRuns, 
    vl_tightsubplot(3, numRuns, i, 'margin', 0.01); imagesc(info.resRand{i}.imsyn); axis image off; 
    vl_tightsubplot(3, numRuns, numRuns+i, 'margin', 0.01); imagesc(info.resQuilt{i}.init); axis image off;
    vl_tightsubplot(3, numRuns, 2*numRuns+i, 'margin', 0.01); imagesc(info.resQuilt{i}.imsyn); axis image off;
end
