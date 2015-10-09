function texture_syn(im, nets, varargin)
clear net;
opts = texture_setup();

% Initialize a dagNN from simple network
[net, objectiveString] = init_dag_cnn(im, nets, opts);

% Initialize random image
x = randn(opts.imageSize, 'single')*127;
x = x(:);

if opts.useGPU
  x = gpuArray(x);
end

x = minFunc(@(x) texture_fun(x, net, objectiveString, opts), x, opts.minFunc); 

% Display image
figure(1); clf;
subplot(1,2,1);
imagesc(im); axis image off;
subplot(1,2,2);
x_ = bsxfun(@plus, reshape(x, opts.imageSize), nets.normalization.averageImage) ;
imagesc(vl_imsc(x_));
axis image off;
