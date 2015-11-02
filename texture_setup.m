function opts = texture_setup(varargin)
% Options for texture synthesis
opts.imageSize = [224 224 3];
opts.TVbeta = 2;
opts.lambdaTV = 1e-6;
opts.beta = 2;
opts.lambdaLb = 0;

% List of layers for content representation
%opts.contentLayer = {'relu4_2'};
%opts.contentLayerWeights = [1];
opts.contentLayer = {};
opts.contentLayerWeights = [];

% List of layers for texture representation
opts.textureLayer = {'relu1_1','relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'};
opts.textureLayerWeights = [1 1 1 1 1]*1;
%opts.textureLayer = {'relu1_1'};
%opts.textureLayerWeights = [1];

% List of layers for texture attribute representation
opts.attributeLayer = {'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'};
opts.attributeLayerWeights = [1 1 1 1]*2000;
%opts.attributeLayer = {};
%opts.attributeLayerWeights = [];

opts.attributeDir = 'data/dtd-weights';
opts.attributeTarget = {'honeycombed'};
opts.attributeTargetWeights = [1];
opts.useGPU = true;
opts.clipPrctile = 1; % Use this to determine min/max

% Options for texture initialization
opts.textureInit = 'rand'; % {'rand', 'quilt'};
opts.randScale = 127;
opts.quiltPatchSize = 24;
opts.quiltOverlap = 2;

% minFunc options (no pesky learning rates for L-BFGS)
opts.maxIter = 100;
opts.minFunc.display = 'iter';
opts.minFunc.useMex = false;

% Parse additional options
opts = vl_argparse(opts, varargin{:});
opts.minFunc.maxIter = opts.maxIter;
