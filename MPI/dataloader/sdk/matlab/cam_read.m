function [M,N] = cam_read(filename)
% Read intrinsic and extrinsic camera matrices M, N from FILENAME.

% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.

TAG_FLOAT = 202021.25;  % check for this when READING the file

if nargout < 2
    error('cam_read: Error. Two outputs required.');
end

if isempty(filename)
    error('cam_read: empty filename.');
end

idx = findstr(filename, '.');
idx = idx(end);

if length(filename(idx:end)) == 1
    error('cam_read: extension required in filename %s', filename);
end;

if strcmp(filename(idx:end), '.cam') ~= 1    
    error('cam_read: filename %s should have extension ''.cam''', filename);
end;

fid = fopen(filename, 'r');
if (fid < 0)
    error('cam_read: could not open %s', filename);
end;

% Check correct endian-ness.
tag     = fread(fid, 1, 'float32');
if (tag ~= TAG_FLOAT)
    error('cam_read(%s): wrong tag (possibly due to big-endian machine?)', filename);
end;

% Read intrinsic matrix
M_ = fread(fid, 9, 'double');
M = reshape(M_, [3,3])';

N_ = fread(fid, 12, 'double');
N = reshape(N_, [4,3])';

fclose(fid);

