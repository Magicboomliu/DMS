function segmentation = segmentation_read(filename)
% Read a segmentation image FILENAME into label image SEGMENTATION.

% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.

I_raw = imread(filename);

I_r = int32(I_raw(:,:,1));
I_g = int32(I_raw(:,:,2));
I_b = int32(I_raw(:,:,3));

segmentation = (I_r * 256 + I_g) * 256 + I_b; 

