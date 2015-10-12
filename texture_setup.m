function opts = texture_setup(varargin)
% Options for texture synthesis
opts.imageSize = [224 224 3];
opts.TVbeta = 2;
opts.lambdaTV = 1e-6;
opts.beta = 2;
opts.lambdaLb = 0;
opts.textureLayer = {'relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'};
opts.textureLayerWeights = [1 1 1 1 1];
opts.useGPU = false;
opts.clipPrctile = 0.1; % Use this to determine the min and max
                        % [range: 0-100]

% Options for texture initialization
opts.textureInit = 'rand'; % {'rand', 'quilt'};
opts.rand.scale = 127;
opts.quilt.patchSize = 24;
opts.quilt.overlap = 2;

% minFunc options (no pesky learning rates)
opts.minFunc.display = 'iter';
opts.minFunc.maxIter = 1000;
opts.minFunc.useMex = false;

% Parse additional options
opts = vl_argparse(opts, varargin);
