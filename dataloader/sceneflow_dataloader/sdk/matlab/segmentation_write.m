function segmentation_write(filename,segmentation)
% Write label image SEGMENTATION to FILENAME.
%
%
% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.


if min(segmentation(:)) < 0
    error(['segmentation_write: Invalid label value ' num2str(min(segmentation(:)))])
end

[h,w] = size(segmentation);

d_r = uint8(floor(segmentation / 256^2));
d_g = uint8(floor(mod(segmentation, 256^2) / 256));
d_b = uint8(floor(mod(segmentation, 256)));

out = uint8(zeros(h,w,3));

out(:,:,1) = d_r;
out(:,:,2) = d_g;
out(:,:,3) = d_b;

imwrite(out, filename);

