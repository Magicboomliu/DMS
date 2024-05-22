addpath('../matlab/');

% Generate data
u = single(rand(500,1000));
v = single(rand(500,1000));
u = u * 100;
v = v * 100;

I = rand(3,3);
E = rand(3,4);

% Test flow
uv = cat(3,u,v);
flow_write('./_test_flow.flo',uv);
uv_ = flow_read('./_test_flow.flo');
error_uv = sqrt(sum((uv-uv_).^2,3));
disp(['Flow error = ' num2str(mean(error_uv(:)))])


% Test depth
depth_write('./_test_depth.dpt',u);
u_ = depth_read('./_test_depth.dpt');
error = mean(abs(u(:)-u_(:)));
disp(['Depth error = ' num2str(mean(error))]);

% Test disparity
disparity_write('./_test_disparity.png',v);
v_ = disparity_read('./_test_disparity.png');
error = mean(abs(v(:)-v_(:)));
disp(['Disparity error = ' num2str(error) ]);

% Test cam matrices
cam_write('./_test_cam.cam',I,E);
[i_,e_] = cam_read('./_test_cam.cam');
error_e = mean(abs(E(:)-e_(:)));
error_i = mean(abs(I(:)-i_(:)));
disp(['Error int: ' num2str(error_i) '. Error ext: ' num2str(error_e)])


% Test segmentation
seg = int32(u);
segmentation_write('./_test_segmentation.png',seg);
seg_ = segmentation_read('./_test_segmentation.png');
error_seg = mean(abs(seg(:)-seg_(:)));
disp(['Error segmentation = ' num2str(error_seg) ]);

% Test and display some real data
FLOWFILE = '../../basic/data/out/temple_2/flow/frame_0001.flo';
DEPTHFILE = '../../stereo/data/package/training/depth/temple_2/frame_0001.dpt';
DISPFILE = '../../stereo/data/package/training/disparities/temple_2/frame_0001.png';
CAMFILE = '../../stereo/data/package/training/camdata_left/temple_2/frame_0001.cam';
SEGFILE = '../../segmentation/data/package/training/segmentation/temple_2/frame_0001.png';

% Load data
uv = flow_read(FLOWFILE);
depth = depth_read(DEPTHFILE);
disp = disparity_read(DISPFILE);
[I,E] = cam_read(CAMFILE);
seg = segmentation_read(SEGFILE);

u = uv(:,:,1);
v = uv(:,:,2);

% Display data
figure()
subplot(3,2,1)
imshow(u,[min(u(:)) max(u(:))])
title('u')
subplot(3,2,2)
imshow(v,[min(v(:)) max(v(:))])
title('v')
subplot(3,2,3)
imshow(depth,[min(depth(:)) max(depth(:))])
title('depth')
subplot(3,2,4)
imshow(disp,[min(disp(:)) max(disp(:))])
title('disparity')
subplot(3,2,5)
imshow(seg,[min(seg(:)) max(seg(:))]);
title('Segmentation')

I
E

