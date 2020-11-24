% function [g1,recons_err]=fn_phase_retrieval_GPU(sautocorr,beta_start,beta_step,beta_stop,N_iter,init_guess,useGPU)

% run e.g.:
% [g1,recons_err]=fn_phase_retrieval(sautocorr,2,-0.04,0,30,rand(size(sautocorr)));

function [g1,recons_err,recons_err2]=BasicPhaseRetrieval(sautocorr_temp,beta_start,beta_step,beta_stop,N_iter,init_guess,useGPU)
if useGPU
    sautocorr=gpuArray(single(sautocorr_temp));
    cur_err=gpuArray(single(zeros(1,N_iter*(length(2:-0.05:0)+1))));
    g1=gpuArray(single(init_guess)); % random initial guess
else
    sautocorr=single(sautocorr_temp);
    cur_err=single(zeros(1,N_iter*(length(2:-0.05:0)+1)));
    g1=single(init_guess); % random initial guess
end

ii=0;
BETAS = [beta_start:beta_step:beta_stop];
for ibeta=1:length(BETAS)
    beta = BETAS(ibeta);
    for iter=1:N_iter
        ii=ii+1;
        G_uv=fft2(g1);
        g1_tag=real(ifft2(sautocorr.*G_uv./abs(G_uv)));
        g1=g1_tag.*(g1_tag>=0) + (g1_tag<0).*(g1 - beta*g1_tag); % my implementatio according to supp.mat
    end
    %fprintf('computing.... %0.1f percents\n',ibeta/length(BETAS)*100);
end

% error reduction part:
for iter=1:N_iter
    ii=ii+1;
    G_uv=fft2(g1);
    g1_tag=real(ifft2(sautocorr.*G_uv./abs(G_uv)));
    g1=g1_tag.*(g1_tag>=0);% + (g1_tag<0).*(g1 - beta*g1_tag); % my implementatio according to supp.mat
end

recons_err=mean(mean((abs(fft2(g1))-sautocorr).^2));
recons_err2=sqrt(mean(mean((abs(fft2(g1)).^2-sautocorr.^2).^2)));

