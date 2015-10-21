% Demo for texture synthesis
% Setup directories
setup;

% Prepare image
im = imread('data/textures/Camouflage0011_S.png');
%im = imread('food.jpg');
im = imresize(im, [224 224]);

% Load network
if ~exist('net', 'var')
    net = load('data/models/imagenet-vgg-verydeep-16.mat');
end

% Run texture synthesis
tic;
imsyn = texture_syn(im, net);
toc;

% Display synthesized texture image
figure(1); clf;
subplot(1,2,1);imagesc(im); axis image off;
title('input');
subplot(1,2,2);imagesc(imsyn); axis image off;
title('synthesized'); set(gca, 'Color', [1 1 1]);