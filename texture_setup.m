function opts = texture_setup(varargin)

opts.learningRate = 0.001*[...
                          ones(1,10000), ...
                      0.1 * ones(1,500), ...
                     0.01 * ones(1,50000), ...
                    0.001 * ones(1,20000), ...
                   0.0001 * ones(1,10000) ] ;

opts.imageSize = [224 224 3];               
opts.TVbeta = 2;
opts.momentum  = 0.9;
opts.lambdaTV = 0;
opts.beta = 3;
opts.lambdaL2 = 0;
opts.textureLayer = {'relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'};
opts.textureLayerWeights = [1 1 1 1 1];

opts = vl_argparse(opts, varargin);