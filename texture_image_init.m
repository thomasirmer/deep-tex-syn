function x = texture_image_init(opts, im)
switch opts.textureInit 
  case 'rand'
    % Randomly initialize the image with Gaussian noise
    x = randn(opts.imageSize, 'single')*opts.rand.scale;
  case 'quilt'
    % Texture quilting
    if nargin < 2,
        error('texture_image_init: no image specified for quilting');
    end
    x = quilt_texture(im, opts);
  otherwise
    error('invalid texture init option.');
end

function x = quilt_texture(im, opts)
n = ceil(max(opts.imageSize)/(opts.quilt.patchSize-opts.quilt.overlap));
x = imagequilt(im, opts.quilt.patchSize, n, opts.quilt.overlap);
x = single(x(1:opts.imageSize(1), 1:opts.imageSize(2), :));