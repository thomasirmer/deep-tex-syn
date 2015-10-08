nets = load('data/models/imagenet-vgg-verydeep-16.mat');
%nets = load('data/models/imagenet-caffe-alex.mat');

% Remove some layers
nets.layers = nets.layers(1:1);

% Add a bilinear pool layer
netLayer.type = 'bilinearpool';%
netLayer.name = 'bpool1';
nets.layers{end+1} = netLayer;

% Convert simplenn to dagNN
net = dagnn.DagNN();
net = net.fromSimpleNN(nets);

% Add a l2 loss layer
net.addLayer('t1error', dagnn.Loss('loss', 'l2'), {'x2','t1'}, 't1error') ;

% Take image and compute responses
im = imread('food.jpg');
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, nets.normalization.imageSize(1:2)) ;
im_ = im_ - nets.normalization.averageImage ;

% Run the CNN
ress = vl_simplenn(nets, im_) ;
preds = ress(end).x;

% Run the DAG NN
net.eval({'x0', im_}) ;
pred = net.vars(net.getVarIndex('x2')).value ;

useGPU = true;

% Image synthesis
x0_sigma = 2.7098e+04;
x = randn(size(im_), 'single') ;
x = x / norm(x(:)) * x0_sigma  ; 


if useGPU
  x = gpuArray(x);
  pred = gpuArray(pred);
  net.move('gpu');
end


opts.learningRate = 0.001*[...
		  ones(1,100000), ...
		  0.1 * ones(1,50000), ...
		  0.01 * ones(1,50000), ...
		  0.001 * ones(1,20000), ...
		  0.0001 * ones(1,10000) ] ;

opts.TVbeta = 2;
opts.momentum  = 0.09;
opts.lambdaTV = 0;
opts.beta = 3;
opts.lambdaL2 = 0;

x_momentum = zeros(size(x), 'single') ;
y0_sigma = norm(squeeze(pred(:)));

net.vars(net.getVarIndex('t1error')).precious = true;
net.vars(net.getVarIndex('x2')).precious = true;

for iter = 1:length(opts.learningRate), 
    inputs = {'x0', x, 't1', pred};
    net.eval(inputs, {'t1error', 1});
    objective=net.vars(net.getVarIndex('t1error')).value;
    dzdx = net.vars(net.getVarIndex('x0')).der;

    [r_,dr_] = tv(x,opts.TVbeta) ;
    dr = dzdx + opts.lambdaTV/2 * dr_ ;

    dr_ = opts.beta * x.^(opts.beta-1) ;
    dr = dr + opts.lambdaL2/2 * dr_ ;
    
    lr = opts.learningRate(iter);
    x_momentum = opts.momentum * x_momentum - (lr*x0_sigma^2/y0_sigma^2) * dr;
    x = x + x_momentum ;
    titleStr = sprintf('iter: %04i objective: %f', iter, objective);
    fprintf('%s\n', titleStr);
    if mod(iter, 500) == 0,
      figure(1); clf;
      subplot(1,2,1);
      imagesc(im); axis image off;
      subplot(1,2,2);
      x_ = bsxfun(@plus, x, nets.normalization.averageImage) ;
      imagesc(vl_imsc(x_));
      %imagesc(x_); colormap gray;
      axis image off;
      drawnow;
      pause(1);
    end
end
