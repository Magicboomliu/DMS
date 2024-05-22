function depth_write(filename, depth)
% Write depth array DEPTH into FILENAME. 

% Adapted from Deqing Sun and Daniel Scharstein's optical
% flow code.

% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.


TAG_STRING = 'PIEH';    % use this when WRITING the file

% sanity check
if isempty(filename) == 1
    error('depth_write: empty filename');
end;

idx = findstr(filename, '.');
idx = idx(end);             % in case './xxx/xxx.dpt'

if length(filename(idx:end)) == 1
    error('depth_write: extension required in filename %s', filename);
end;

if strcmp(filename(idx:end), '.dpt') ~= 1    
    error('depth_write: filename %s should have extension ''.dpt''', filename);
end;

[height width] = size(depth);

fid = fopen(filename, 'w');
if (fid < 0)
    error('depth_write: could not open %s', filename);
end;

% write the header
fwrite(fid, TAG_STRING); 
fwrite(fid, width, 'int32');
fwrite(fid, height, 'int32');

% arrange into matrix form
tmp = zeros(height, width);
tmp(:) = depth;
tmp = tmp';

fwrite(fid,tmp,'float32');
fclose(fid);

