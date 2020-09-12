clear all;

pathl='E:\glmdata\glmatlab\testdata\';
fileForm = '*.mat';
files1 = dir(fullfile(pathl,fileForm)); 
len1 = size(files1,1);
for i=1:len1
    filename1 = strcat(pathl,files1(i).name);
    sidelength = 64;%Í¼Æ¬³ß´ç
    subrate =0.25;
    epsilon = 0.01;
    eval(['load ' filename1])
    N = sidelength^2;
    y = a(1:(round(N*subrate)));
    y=y';
    M = length(y);
    load 'E:\glmdata\glmatlab\4096.mat'%±àÂëmat
    Phi = iiss(1:M,:);
    x0 = Phi\y;
    estIm = tvqc_logbarrier(x0, Phi, [], y, epsilon,1e-4,2);
    xtvqc = reshape(estIm,sqrt(N),sqrt(N));
    xtvqc = flipud(xtvqc);
    xtvqc = (xtvqc-min(min(xtvqc)))/(max(max(xtvqc))-min(min(xtvqc)));
%     figure,imagesc(xtvqc),colormap(gray),axis image
%     autocorr = fftXcorr2(xtvqc,1);
%     figure, imshow(autocorr,[]);
    
    
end





