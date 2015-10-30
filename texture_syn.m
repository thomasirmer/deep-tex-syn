function res = texture_syn(im, nets, varargin)
clear net;
opts = texture_setup(varargin{:});

% Initialize a dagNN from simple network
[net, objectiveString] = init_dag_cnn(im, nets, opts);

% meanRGB value
meanRGB = mean(mean(nets.normalization.averageImage));

% Initialize random image
x = texture_image_init(opts, im, meanRGB); x = x(:);

% Book keeping
res.init = bsxfun(@plus, reshape(x, opts.imageSize), meanRGB)/255;

% Move input to GPU if needed (note: net is on the GPU already)
if opts.useGPU
  x = gpuArray(x);
end

% Run L-BFGS optimization 
[x,~,~,info] = minFunc(@(x) texture_fun(x, net, objectiveString, opts), x, opts.minFunc); 

% Un-normalizate the image
x_ = bsxfun(@plus, reshape(x, opts.imageSize), meanRGB);

% Resale it to the valid range
xrange = prctile(x_(:),[opts.clipPrctile 100-opts.clipPrctile]);
imsyn = vl_imsc(min(max(x_,xrange(1)), xrange(2)));

% Write out the outputs for book keeping
res.imsyn = imsyn;
res.x_ = x_;
res.opts = opts;
res.info = info;
res.objectiveString = objectiveString;