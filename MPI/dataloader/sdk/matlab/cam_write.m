function cam_write(filename,M,N)
% Write intrinsic and extrinsic camera matrices M,N to FILENAME.

% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.

TAG_STRING = 'PIEH';    % use this when WRITING the file

% sanity check
if isempty(filename) == 1
    error('cam_write: empty filename');
end;

if nargin < 3
    error('cam_write: Three inputs required.');
end

idx = findstr(filename, '.');
idx = idx(end);             % in case './xxx/xxx.cam'

if length(filename(idx:end)) == 1
    error('cam_write: extension required in filename %s', filename);
end;

if strcmp(filename(idx:end), '.cam') ~= 1
    error('cam_write: filename %s should have extension ''.cam''', filename);
end;

fid = fopen(filename, 'w');
if (fid < 0)
    error('cam_write: could not open %s', filename);
end;

% write the header
fwrite(fid, TAG_STRING);
fwrite(fid, M', 'double');
fwrite(fid, N', 'double');

fclose(fid);

