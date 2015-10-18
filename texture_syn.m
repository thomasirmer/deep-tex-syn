function imsyn = texture_syn(im, nets, varargin)
clear net;
opts = texture_setup();

% Initialize a dagNN from simple network
[net, objectiveString] = init_dag_cnn(im, nets, opts);

% Initialize random image
x = texture_image_init(opts, im); x = x(:);

% Move input to GPU if needed (note: net is on the GPU already)
if opts.useGPU
  x = gpuArray(x);
end

% Run L-BFGS optimization 
x = minFunc(@(x) texture_fun(x, net, objectiveString, opts), ...
            x, opts.minFunc); 

% Un-normalizate the image
x_ = bsxfun(@plus, reshape(x, opts.imageSize), ...
            mean(mean(nets.normalization.averageImage))) ;

% Resale it to the valid range
xrange = prctile(x_(:),[opts.clipPrctile 100-opts.clipPrctile]);
imsyn = vl_imsc(min(max(x_,xrange(1)), xrange(2)));