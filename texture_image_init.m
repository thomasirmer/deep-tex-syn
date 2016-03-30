function x = texture_image_init(opts, im, meanRGB)
switch opts.textureInit 
  case 'rand'
    % Randomly initialize the image with Gaussian noise
    x = randn(opts.imageSize, 'single')*opts.randScale;
  case 'quilt'
    % Texture quilting
    if nargin < 2,
        error('texture_image_init: no image specified for quilting');
    end
    x = quilt_texture(im, opts);
    x = bsxfun(@minus, x, meanRGB);
  case 'same'
    % Same as the input image
    x = single(im);
    x = bsxfun(@minus, x, meanRGB);
  case 'transfer-quilt'
    % Texture transfer with quilting
    if nargin < 2,
        error('texture_image_init: no image specified for quilting');
    end
    x = transfer_texture(im, opts);
    x = bsxfun(@minus, x, meanRGB);
    
  otherwise
    error('invalid texture init option.');
end

function x = quilt_texture(im, opts)
n = ceil(max(opts.imageSize)/(opts.quiltPatchSize-opts.quiltOverlap));
x = imagequilt(im, opts.quiltPatchSize, n, opts.quiltOverlap);
x = single(x(1:opts.imageSize(1), 1:opts.imageSize(2), :));



function x = transfer_texture(im, opts)
imdb = dtd_get_database(opts.attributeDataset);
classIdx = find(strcmp(imdb.classes.name, opts.attributeTarget{1}));
imIdx = find(imdb.images.label==classIdx);
im2 = imread(fullfile(imdb.imageDir, imdb.images.name{imIdx(1)}));

n = ceil(max(opts.imageSize)/(opts.quiltPatchSize-opts.quiltOverlap));
x = quilttransfer(im2, opts.quiltPatchSize, n, opts.quiltOverlap, 0.002, opts.transferalpha, im);
x = single(x(1:opts.imageSize(1), 1:opts.imageSize(2), :));
