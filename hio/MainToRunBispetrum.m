clear;
close all;
set(0,'DefaultFigureWindowStyle','docked');

Large_Speckle_intensity = double(imread('Digit 4 experimental data.tif'));
Large_Speckle_intensity = flip(Large_Speckle_intensity,1);
Large_Speckle_intensity = awgn(Large_Speckle_intensity,Inf,'measured');
[N_Speckle_1,N_Speckle_2] = size(Large_Speckle_intensity);
MyMap = 'jet';
% projection_theta = 60;
projection_theta = 0:10:179;

A1 = double(imread('Digit 4-240.png'));
% Obj = double(rgb2gray(imread('C:\Users\40716\Desktop\SpeckleTest\single speckle pattern for discussion\manuscript prepare\Experimental data-Tengfei\Edmund分辨率靶-1列数字1 4 5 6\Digit 4 for recontrution error.png')));    % For comparing error
A1FFT = fftshift(fft2(ifftshift(A1)));
R = radon(A1,projection_theta);  % 1D projections of 2D image with Radon transform  
[projectionLength,projectionNum] = size(R);
RealLengthCurrentObj = projectionLength-1;

% R_choose = 2:(projectionLength-2);
R_choose = 1:RealLengthCurrentObj;
R = R(R_choose,:);
R_Recover = zeros(RealLengthCurrentObj,projectionNum);

sizeChoose = 0.5;
% envelopeDimension = sizeChoose*7*N;
envelopeDimension = 240;   % related to the size of the object

% Note: selection of parameters "envelopeDimension" and "sigma" is very important.
% Except for the principle metioned above, the parameters should make the 
% envelope in one frame as large as possible, to ensure the Fourier transform
% of the Gaussian window function ~delta function (small enouhg).
% Too small envope would smoothen the bispectrum phase information (lose
% phase information of object).

sigma = 50; % sigma~(2D-3D) D is size of object, larger sigma ensure more information, but introduce more computational complexity  
x = (-(floor(envelopeDimension/2)-1):(floor(envelopeDimension/2)));
[X,Y] = meshgrid(x);
envelope = exp(-(X.^2+Y.^2)/(sigma^2));   
% envelope = window2(envelopeDimension,envelopeDimension,@hamming);
interval = envelopeDimension;
overlap_ratio = 0.85;
overlap_size = interval*overlap_ratio;


%% filter begin
Large_Speckle_intensity_FFT = fftshift(fft2((Large_Speckle_intensity)));
LowPass_Filter = zeros(N_Speckle_1,N_Speckle_2);
Filter_Length = 15;
LowPass_Filter((floor(N_Speckle_1/2)-Filter_Length):(floor(N_Speckle_1/2)+Filter_Length-1),(floor(N_Speckle_2/2)-Filter_Length):(floor(N_Speckle_2/2)+Filter_Length-1)) = 1;
Large_Speckle_intensity_FFT = Large_Speckle_intensity_FFT.*LowPass_Filter;
Large_Speckle_intensity_LowPassVer = abs(ifft2(ifftshift(Large_Speckle_intensity_FFT)));
% smooth_fac=60; % better be larger than acorr_size (too small - objects edges disappear), too large- bad reconst?
% Specklesmooth=conv2fft(Large_Speckle_intensity,ones(smooth_fac),'same');
Large_Speckle_intensity_New = Large_Speckle_intensity./Large_Speckle_intensity_LowPassVer;
h = fspecial('gaussian',[4 4],0.5);
Large_Speckle_intensity = imfilter(Large_Speckle_intensity_New,h,'replicate');
% Large_Speckle_intensity = Large_Speckle_intensity(100:2100,:);
[N_Speckle_1,N_Speckle_2] = size(Large_Speckle_intensity);
%% filter end

% %%%%%%%%%%%%TEST%%%%%%%%%%%%%%%%
% SpecklePattern = load('LargeSpeckle.mat');
% Large_Speckle_intensity = SpecklePattern.Large_Speckle_intensity;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



Large_Speckle_intensity_FFT_intensity = abs(fft2(fftshift(Large_Speckle_intensity))).^2;
Large_Speckle_intensity_AutoCorr = ifftshift(ifft2((Large_Speckle_intensity_FFT_intensity)));

% num_sqrt = floor(N_Speckle/interval);
% num_sqrt = 43;
num_sqrt_1 = floor(1+(N_Speckle_1-interval)/((1-overlap_ratio)*interval));
num_sqrt_2 = floor(1+(N_Speckle_2-interval)/((1-overlap_ratio)*interval));
num = num_sqrt_1*num_sqrt_2;
% A = zeros(N_Speckle,N_Speckle);
% Bispect = zeros(N_Speckle,N_Speckle);

subspeckle = cell(num_sqrt_1,num_sqrt_2);

%% division begin
for i = 1:num_sqrt_1
    for j = 1:num_sqrt_2
        subspeckle{i,j} = (Large_Speckle_intensity((uint16((i-1)*(1-overlap_ratio)*interval+1):uint16(interval*i-overlap_ratio*(i-1)*interval)),...
            (uint16((j-1)*(1-overlap_ratio)*interval+1):uint16(interval*j-overlap_ratio*(j-1)*interval))));
%         I_FFT_intensity = abs(fft2(fftshift(subspeckle{i,j}))).^2;
%         I_AutoCorr = ifftshift(ifft2((I_FFT_intensity)));
%         figure(1);
%         set(gcf,'outerposition',get(0,'screensize'));
%         subplot(121);imshow(subspeckle{i,j},[]);colormap hot;colorbar
%         subplot(122);imshow(I_AutoCorr,[]);colormap hot;colorbar
%         pause(1);
    end
end
subspeckle = subspeckle(:);
%% division end


fprintf('End of the division process.\n\n');
fprintf('Start to projection process...\n\n');

for q = 1:num

    Speckle_intensity = subspeckle{q};
%     Speckle_intensity = Speckle_intensity(1:180,1:180);
    Speckle_intensity = Speckle_intensity.*envelope;       % filter in the spatial domain with Gaussian window, and its kernel
                                                           % should be between 2D and 3D, where D is the size of object
                                                                                                                
%     Speckle_Intensity_FFT = abs(fft2(fftshift(Speckle_intensity))).^2;
%     Speckle_Intensity_AutoCorr = ifftshift(ifft2((Speckle_Intensity_FFT)));

    R_Speckle = radon(Speckle_intensity,projection_theta);
    [projectionLength_Speckle,~] = size(R_Speckle);
    RealLengthCurrentSpeckle = projectionLength_Speckle-1;         
%     R_choose_Speckle = 2:(projectionLength_Speckle-2);              
    R_choose_Speckle = 1:RealLengthCurrentSpeckle;                  
    R_Speckle = R_Speckle(R_choose_Speckle,:);
    R_Speckle_Store(:,:,q) = R_Speckle;
end

fprintf('End of the projection process.\n\n');
fprintf('Start to reconstruction.\n\n');

offset = floor(RealLengthCurrentSpeckle/2);

Object_ReSpeckleProjFFT = zeros(RealLengthCurrentSpeckle,projectionNum);
R_Recover_Speckel = zeros(RealLengthCurrentSpeckle,projectionNum);


% filter = fspecial('gaussian',[5 5],1);
nfft = RealLengthCurrentSpeckle;
Wind = 1;
nsamp = RealLengthCurrentSpeckle;
overlap = 0;


for projection = 1:projectionNum
    
    fprintf('Start of No.%d projection, %d projections in total.\n\n',projection,projectionNum);
    
    if nfft>=nsamp
        Initial_Bspec_1 = zeros(nfft,nfft);
        Initial_Bspec_2 = zeros(nfft,nfft);
        Initial_Bspec_3 = zeros(nfft,nfft);
    else
        nfft1 = 2^nextpow2(nsamp);
        Initial_Bspec_1 = zeros(nfft1,nfft1);
        Initial_Bspec_2 = zeros(nfft1,nfft1);
        Initial_Bspec_3 = zeros(nfft1,nfft1);
    end
    
    Bspec_Speckle = Initial_Bspec_2;
    currentObj = R(:,projection)';
%     Temp = zeros(1,RealLengthCurrentObj);
%     Temp(1:end-1) = currentObj;
%     currentObj = Temp;
    
    [Bspec,waxis] = Calbispectrum2D(currentObj,nfft,Wind,nsamp,overlap,Initial_Bspec_1);
    Bspec = rot90(Bspec);
    Bspec = Bspec';
    TripCorr = fftshift(fft2(Bspec));
    
    amplitude_esti = zeros(1,RealLengthCurrentSpeckle);
    
    for Frame = 1:num
        
        currentSpeckle_Radon = R_Speckle_Store(:,:,Frame);
%         currentSpeckle_Radon = R_Speckle;
        currentSpeckle_Projection = currentSpeckle_Radon(:,projection)';
        [Bspec_Speckle_1,waxis_Speckle] = Calbispectrum2D(currentSpeckle_Projection,nfft,Wind,nsamp,overlap,Initial_Bspec_2);
        Bspec_Speckle = Bspec_Speckle + Bspec_Speckle_1;
        
        fft_currentSpeckle_Projection = fft(currentSpeckle_Projection);
%         fft_currentSpeckle_Projection = fft(ifftshift(currentSpeckle_Projection));
        amplitude_esti=amplitude_esti+abs(fft_currentSpeckle_Projection).*abs(fft_currentSpeckle_Projection);   % unaccurate estimatation of amplitude and with
                                                                                                                % strong background noise    
    end

    Bspec_Speckle = Bspec_Speckle/num;
    Bspec_Speckle = rot90(Bspec_Speckle);
    Bspec_Speckle = Bspec_Speckle';
    TripCorr_Speckel = fftshift(fft2(Bspec_Speckle));
    amplitude_esti = amplitude_esti/num;
%     amplitude_esti = amplitude_esti - 3e20;
    
    ObjPhase_Recover = RecursiveProcess(Bspec,RealLengthCurrentObj);
    ObjPhase_Recover_Speckle = RecursiveProcess(Bspec_Speckle,RealLengthCurrentSpeckle);
    Object_ReSpeckleProjFFT(:,projection) = cos(ObjPhase_Recover_Speckle)+1i.*sin(ObjPhase_Recover_Speckle);
%     Object_ReSpeckleProjFFT(:,projection) = Object_ReSpeckleProjFFT(:,projection).*exp(1j*2*pi*linspace(0,(RealLengthCurrentSpeckle-1),RealLengthCurrentSpeckle)'*offset/RealLengthCurrentSpeckle);
%     Object_ReSpeckleProjFFT(:,projection) = fftshift(Object_ReSpeckleProjFFT(:,projection));

    ObjPhase_True = angle(fft(currentObj));
    Object_TruePhaseRecover = cos(ObjPhase_True)+1i.*sin(ObjPhase_True);
    Object_TruePhaseRecover = ifft(Object_TruePhaseRecover);
    
    Object_RePhaseRecover = cos(ObjPhase_Recover)+1i.*sin(ObjPhase_Recover);
    Object_RePhaseRecover = ifft(Object_RePhaseRecover);
    
    Object_Recover = abs(fftshift(fft(currentObj))).*cos(ObjPhase_Recover)+abs(fftshift(fft(currentObj))).*1i.*sin(ObjPhase_Recover);
    Object_Recover = ifftshift(ifft(Object_Recover));
    Object_Recover_amplitude = abs(Object_Recover);
    Object_Recover_amplitude = Object_Recover_amplitude';
    R_Recover(:,projection) = Object_Recover_amplitude;
    
%     Object_Recover_Speckle = fftshift(sqrt(amplitude_esti)).*cos(ObjPhase_Recover_Speckle)+fftshift(sqrt(amplitude_esti)).*1i.*sin(ObjPhase_Recover_Speckle);
    Object_Recover_Speckle = abs(fftshift(fft(currentObj))).*cos(ObjPhase_Recover_Speckle)+abs(fftshift(fft(currentObj))).*1i.*sin(ObjPhase_Recover_Speckle);
    Object_Recover_Speckle = ifftshift(ifft(Object_Recover_Speckle));
    Object_Recover_amplitude_Speckle = abs(Object_Recover_Speckle);
    Object_Recover_amplitude_Speckle = Object_Recover_amplitude_Speckle';
    R_Recover_Speckel(:,projection) = Object_Recover_amplitude_Speckle;
    
    
%     figure;subplot(121);plot(ObjPhase_True,'r.');subplot(122);plot(ObjPhase_Recover,'r.');
%     
%     figure(2);
%     subplot(221);plot(currentObj);
%     subplot(222);imshow(abs(Bspec),[]);colormap(jet);
%     subplot(223);imshow(abs(TripCorr),[]);colormap(jet);
%     subplot(224);imshow(angle(Bspec),[]);colormap(jet);colorbar;
%     figure(3);
%     subplot(221);plot(currentSpeckle_Projection);
%     subplot(222);imshow(abs(Bspec_Speckle),[]);colormap(jet);
%     subplot(223);imshow(abs(TripCorr_Speckel),[]);colormap(jet);
%     subplot(224);imshow(angle(Bspec_Speckle),[]);colormap(jet);colorbar;
%     pause(0.5);
%     title3 = strcat('TruePhaseRecover','   ',num2str(projection));
%     title4 = strcat('RePhaseRecover','   ',num2str(projection));
%     figure;
%     subplot(121);plot(abs(Object_TruePhaseRecover));title(title3);
%     subplot(122);plot(abs(Object_RePhaseRecover));title(title4);
%     pause(0.1);
%     title1 = strcat('Current Projection','   ',num2str(projection));
%     title2 = strcat('Reconstruction','   ',num2str(projection));
%     figure;
%     subplot(121);plot(currentObj);title(title1);
%     subplot(122);plot(abs(Object_Recover));title(title2);
%     pause(0.1);
%     figure;
%     subplot(121);plot(abs(Object_TruePhaseRecover));title(title3);
%     subplot(122);plot(abs(Object_RePhaseRecover_Speckle));title(title4);
%     pause(0.1);

    title1 = strcat('Current Projection','   ',num2str(projection));
    title2 = strcat('Reconstruction','   ',num2str(projection));
    figure(1);
    subplot(121);plot(currentObj);title(title1);
    subplot(122);plot(abs(Object_Recover_Speckle));title(title2);
    pause(0.1);

%     titlename = strcat('No.',num2str(projection),' projection');
%     figure(1);
%     plot(currentObj,'-b');
%     hold on
%     plot(abs(Object_Recover_Speckle),'-r');
%     hold off
%     legend('projection of observed object','projection recovered from speckle pattern');
%     title(titlename);
%     pause(0.1);
end

R_Recover_Image = iradon(R,projection_theta);
R_Recover_Recover_Image = iradon(R_Recover,projection_theta);
R_Recover_Recover_Speckle_Image = iradon(R_Recover_Speckel,projection_theta);



% figure;
% subplot(121);imshow(R,[]);colormap(hot);
% subplot(122);imshow(R_Recover,[]);colormap(hot);
% figure;
% subplot(121);imshow(R,[]);colormap(hot);
% subplot(122);imshow(R_Recover_Speckel,[]);colormap(hot);
% figure;
% subplot(121);imshow(R_Recover_Image,[]);colormap hot;colorbar
% subplot(122);imshow(R_Recover_Recover_Image,[]);colormap hot;colorbar
% figure;
% subplot(121);imshow(R_Recover_Image,[]);colormap hot;colorbar
% subplot(122);imshow(R_Recover_Recover_Speckle_Image,[]);colormap hot;colorbar
% figure;
% imshow(Speckle_intensity,[]);colormap hot;

% R_Recover_Recover_Speckle_Image_FFT = fftshift(fft2(ifftshift(R_Recover_Recover_Speckle_Image.^2)));
% phase = angle(R_Recover_Recover_Speckle_Image_FFT);

% transform the polar coordinate to the Cartesian coordinate
phase = PolarToCartesian(RealLengthCurrentSpeckle,projection_theta,Object_ReSpeckleProjFFT);
[~,LengthSelection] = size(phase);

Large_Speckle_intensity_FFT_intensity = abs(fft2(fftshift(Large_Speckle_intensity))).^2;
Speckle_intensity_large_AutoCorr = ifftshift(ifft2((Large_Speckle_intensity_FFT_intensity)));
MaxValue = max(max(Speckle_intensity_large_AutoCorr));
Sparseradio = 0;
backgroundterm = Sparseradio/(Sparseradio+1)*MaxValue;
% Speckle_intensity_large_AutoCorr = Speckle_intensity_large_AutoCorr-backgroundterm;
% Speckle_intensity_large_AutoCorr = Speckle_intensity_large_AutoCorr-min(Speckle_intensity_large_AutoCorr(:));
[N_AutoCorr_1,N_AutoCorr_2] = size(Speckle_intensity_large_AutoCorr);
Speckle_intensity_large_AutoCorr_Select = Speckle_intensity_large_AutoCorr((floor(N_AutoCorr_1/2)+(-LengthSelection/2:(LengthSelection/2-1))),(floor(N_AutoCorr_2/2)+(-LengthSelection/2:(LengthSelection/2-1))));
% Speckle_intensity_large_AutoCorr_Select = Speckle_intensity_large_AutoCorr((floor(N_AutoCorr_1/2)+(-30:29)),(floor(N_AutoCorr_2/2)+(-30:29)));
Speckle_intensity_large_AutoCorr_Select = Speckle_intensity_large_AutoCorr_Select-min(Speckle_intensity_large_AutoCorr_Select(:));
Modulus_estimation = sqrt(abs(fftshift(fft2(Speckle_intensity_large_AutoCorr_Select))));

fprintf('End of reconstruction!\n');
pause(1);

figure;
subplot(221);imshow(Modulus_estimation,[]);colormap hot;colorbar;title('estiamtion Fourier amplitude from autocorrealtion of speckle');
subplot(222);imshow(phase,[]);colormap hot;colorbar;title('estiamtion Fourier phase from bispectrum of speckle');
subplot(223);imshow(abs(A1FFT),[]);colormap hot;colorbar;title('true Fourier amplitude of observed object');
subplot(224);imshow(angle(A1FFT),[]);colormap hot;colorbar;title('true Fourier phase of observed object');
Recover_FFT = Modulus_estimation.*cos(phase)+1i.*Modulus_estimation.*sin(phase);
% Recover_FFT = cos(phase)+1i.*sin(phase);
Recover = fftshift(ifft2(ifftshift(Recover_FFT)));
Recon = abs(Recover);
% recon_erro = ReconQualityBisp(Obj,Recon)

figure;
subplot(121);imshow(abs(Recover),[]);axis off;colormap hot;colorbar;title('recovered object--amplitude display');
subplot(122);imshow(abs(Recover).^2,[]);axis off;colormap hot;colorbar;title('recovered object--intensity display');
% set(gcf,'color','white');

%% Save data
% save('C:\Users\40716\Desktop\SpeckleTest\single speckle pattern for discussion\manuscript prepare\Experimental data-Tengfei\digit 6 amplitude-48-0.9-18projection-9s.mat','Modulus_estimation');
% save('C:\Users\40716\Desktop\SpeckleTest\single speckle pattern for discussion\manuscript prepare\Experimental data-Tengfei\digit 6 phase-48-0.9-18projection-9s.mat','phase');
