function opts = texture_setup(varargin)
% Options for texture synthesis
opts.imageSize = [224 224 3];
opts.TVbeta = 2;
opts.lambdaTV = 1e-5;
opts.beta = 2;
opts.lambdaLb = 0;
opts.textureLayer = {'relu1_1','relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'};
opts.textureLayerWeights = [1 1 1 1 1];
%opts.textureLayer = {};
%opts.textureLayerWeights = [];
opts.attributeLayer = {'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'};
opts.attributeLayerWeights = [1 1 1 1]*500;
opts.attributeDir = 'data/dtd-weights';
opts.attributeTarget = 'grid';
opts.useGPU = true;
opts.clipPrctile = 1; % Use this to determine the min and max
                        % [range: 0-100]

% Options for texture initialization
opts.textureInit = 'rand'; % {'rand', 'quilt'};
opts.rand.scale = 127;
opts.quilt.patchSize = 24;
opts.quilt.overlap = 2;

% minFunc options (no pesky learning rates)
opts.minFunc.display = 'iter';
opts.minFunc.maxIter = 250;
opts.minFunc.useMex = false;

% Parse additional options
opts = vl_argparse(opts, varargin);
