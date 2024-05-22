function disparity_write(filename,disparity,bitdepth)
% Write disparity data DISPARITY to FILENAME.

% bitdepth can be either 16 or 24, to control how much data to write.
% (default: 16).

% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.


if nargin < 3
    bitdepth = 16;
end

% Clip.
disparity(disparity > 1024) = 1024;
disparity(disparity < 0) = 0;

[h,w] = size(disparity);

d_r = uint8(floor(disparity / 4.0));
d_g = uint8(floor(mod(disparity * (2^6), 256)));
d_b = uint8(floor(mod(disparity * (2^14), 256)));

out = uint8(zeros(h,w,3));

out(:,:,1) = d_r;
out(:,:,2) = d_g;

if bitdepth == 24
    out(:,:,3) = d_b;
end

imwrite(out, filename);

