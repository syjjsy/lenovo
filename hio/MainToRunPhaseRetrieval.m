clear all;
close all;
set(0,'DefaultFigureWindowStyle','docked');

trial = 10;
RecontructionResultStore = cell(1,trial);

New_Speckle = double((imread('Digit 4 experimental data.tif')));
New_Speckle = flip(New_Speckle,1);
New_Speckle = awgn(New_Speckle,20,'measured');

A = double(New_Speckle);
A_FL = fftshift(fft2((A)));
[NN1,NN2] = size(A);
T = zeros(NN1,NN2);
Filter_Length = 15;
T(ceil(NN1/2)+(-Filter_Length:Filter_Length),ceil(NN2/2)+(-Filter_Length:Filter_Length)) = 1;
A_FL = A_FL.*T;
A_FL = abs(ifft2(ifftshift(A_FL)));
A_New = A./A_FL;
h = fspecial('gaussian',[4 4],0.5);
AA = imfilter(A_New,h,'replicate');
% AA = AA(200:2000,200:2400);
AF = abs(fft2(fftshift(AA))).^2;
IAF = ifftshift(ifft2((AF)));


[n1 n2] = size(IAF);
Select_AutocorrelationArea_n1 = 60;
Select_AutocorrelationArea_n2 = 60;
Select_AutoArea = IAF(floor(n1/2)-Select_AutocorrelationArea_n1:floor(n1/2)+Select_AutocorrelationArea_n1-1,floor(n2/2)-Select_AutocorrelationArea_n2:floor(n2/2)+Select_AutocorrelationArea_n2-1);
[value index] = max(IAF(:));
Select_AutoArea = Select_AutoArea - min(Select_AutoArea(:));

Fixed_Amp = sqrt(abs(fftshift(fft2(Select_AutoArea))));
% figure(1);imshow(Fixed_Amp,[]);axis off;colormap hot;
    
for k = 1:trial     

    Reconstruct_Field = BasicPhaseRetrieval(ifftshift(Fixed_Amp),2,-0.04,0,30,rand(size(Select_AutoArea)),0);
    Reconstruct_Image_Inten = abs(Reconstruct_Field).^2;
%     Reconstruct_Image_Inten = mat2gray(Reconstruct_Image_Inten);
%     Reconstruct_Image_Inten = 1 - Reconstruct_Image_Inten;
    Reconstruct_Image_Amp = abs(Reconstruct_Field);
    
    figure;imshow(Reconstruct_Image_Amp,[]);colormap hot;
    pause(0.5);
% %     
%     figure;
%     subplot(121)
%     imshow(Reconstruct_Image_Amp,[]);
%     colormap(hot);
%     subplot(122)
%     imshow(Reconstruct_Image_Inten,[]);
%     colormap(hot);
%     colorbar;
    
    
    RecontructionResultStore{k} = Reconstruct_Image_Amp;
    fprintf('End of No.%d trial\n\n',k);
end

% save('C:\Users\40716\Desktop\SpeckleTest\single speckle pattern for discussion\manuscript prepare\Experimental data-Tengfei\digit 4 phase retrieval-7dB-200 trials.mat','RecontructionResultStore');
% recon_erro = ReconQuality(Object,RecontructionResultStore);
% figure;plot(recon_erro,'*-r');