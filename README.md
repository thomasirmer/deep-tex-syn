# Parametric models for texture
 
This repository contains the code for texture synthesis using CNN models


List of things to do:

	1. Texture experiments:
		1. DTD, FMD, KTH-2b datasets
		2. MIT Indoor, PASCAL VOC 2007
	2. Texture synthesis:
	 	1. Learn weights for various layers using a dataset of texture samples (take multiple crops and learn weights).
	 	2. Comparison of patch-based methods for texture synthesis
	3. Texture modification
		1. Constrained optimization problems with decribable attributes for editing the textures.

## Installing

The code depends on VLFEAT and MatConvNet. Download and install these in the current directories. The code is tested under MatConvNet version `v1-beta15`. You can follow the detailed instructions on the project pages to install these on your local machine. For example here are the steps I followed to install these on a MacBookPro laptop running MATLAB_R2011a. 

To install MatConvNet (w/o GPU support):

	>> git clone git@github.com:vlfeat/matconvnet.git
	>> cd matconvnet
	>> git checkout -n v1-best15
	>> make

Similarly, to intall VLFEAT:

	>> git clone git@github.com:vlfeat/vlfeat.git
	>> cd vlfeat
	>> make ARCH=maci64 MEX=/Applications/MATLAB_R2011a.app/bin/mex


Once installed modify `setup.m` to point to their locations.

