% Setup MatConvNet
run matconvnet/matlab/vl_setupnn.m;

% Setup VLFEAT
run vlfeat/toolbox/vl_setup.m;

% Setup minFunc
addpath(genpath('minFunc/'));

% Setup imagequilt
addpath('imagequilt');

% Setup B-CNN code
addpath('bcnn-extension')

% Setup experiments
addpath('exp/');
