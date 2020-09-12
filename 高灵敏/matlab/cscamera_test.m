% Compressive Imaging Reconstruction Tests
%
% Written by: Marco F. Duarte
%             Digital Signal Processing Group
%             Rice University
%             November 14 2006
%
% Requires l1-Magic, available at http://www.l1-magic.org
% NOTE: The 'Optimization' folder in the l1-Magic toolbox must be in the
% path

% 
% CONFIGURABLE PARAMETERS
%
clear all;
% Test picture name. Options include 'Ball','Mug','Dice','Logo', 'R'
picname = '33';%采图mat
% Picture size (sidelength in pixels). Options include 32, 64, 128
sidelength = 64;%图片尺寸
% Subsampling rate: ratio between M (number of samples) and image size in
% pixels (picsize^2) 压缩
subrate =0.5;
% Optimization parameters 优化参数
epsilon = 0.01;

%
% SCRIPT
%

% Load measurements - Column Vector
eval(['load ' picname '_' num2str(sidelength)])

% Image size
N = sidelength^2;

% Get desired number of samples
y = lista(1:(round(N*subrate)));
% y=y/max(y);
% Number of measurements
M = length(y);

% Load measurement matrix and truncate
% Each row of Phi contains a measurement vector of length N
% and is a raster scan of the 0/1 pattern fed to the DMD
eval(['load CSmask_' num2str(sidelength)])%编码mat

Phi = Phi(1:M,:);

% Set initial solution and run TV minimizer
% Output is in column format, raster scan of reconstructed image
% Initial point: minimum energy
x0 = Phi\y;
estIm = tvqc_logbarrier(x0, Phi, [], y, epsilon,1e-4,2);

% Reshape output; flip upside down
xtvqc = reshape(estIm,sqrt(N),sqrt(N));
xtvqc = flipud(xtvqc);
%Normalize output3
xtvqc = (xtvqc-min(min(xtvqc)))/(max(max(xtvqc))-min(min(xtvqc)));
% Show reconstruction
figure,imagesc(xtvqc),colormap(gray),axis image
autocorr = fftXcorr2(xtvqc,1);
figure, imshow(autocorr,[]);
% save('1013 4096对比1.mat','autocorr');
% save 200_img1yasuolv.mat xtvqc
