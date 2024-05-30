function disparity = disparity_read(filename)
% Read a disparity image FILENAME into data array DISPARITY.

% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.

I_raw = imread(filename);

I_r = double(I_raw(:,:,1));
I_g = double(I_raw(:,:,2));
I_b = double(I_raw(:,:,3));

disparity = I_r * 4 + I_g / (2^6) + I_b / (2^14); 

