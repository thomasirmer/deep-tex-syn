%function texture_syn(im, nets, varargin)
nets = load('data/models/imagenet-vgg-verydeep-16.mat');
clear net;
opts = texture_setup();

% Initialize a dagNN from simple network
[net, objectiveString] = init_dag_cnn(im, nets, opts);

%% Initialize random image
options.display = 'iter';
options.maxFunEvals = 1000;
options.useMex = false;
x0_sigma = 2.7098e+04;
x = randn(opts.imageSize, 'single') ;
x = x / norm(x(:)) * x0_sigma ; 
x = x(:);
x = minFunc(@(x) texture_fun(x, net, objectiveString), x, options); 

% Display image
figure(1); clf;
subplot(1,2,1);
imagesc(im); axis image off;
subplot(1,2,2);
x_ = bsxfun(@plus, reshape(x, opts.imageSize), nets.normalization.averageImage) ;
imagesc(vl_imsc(x_));
axis image off;
