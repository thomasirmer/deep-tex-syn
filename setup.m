% Setup MatConvNet
run matconvnet/matlab/vl_setupnn.m;

% Setup VLFEAT
run vlfeat/toolbox/vl_setup.m;

% Setup minFunc
addpath(genpath('minFunc/'));

% Setup experiments
addpath('exp/');
