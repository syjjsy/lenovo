function Phi = RecursiveProcess(Bispectrum,N)
% This function is used to recover the 1D-object phase from its bispectrum
% with the recursie method.The iterative equation is as follows:
%         ¦×(r) = ¦×(q)+¦×(r-q)-¦Â(r-q,q)
% The main parameters are listed below:

% Bispectrum: the bispectrum of object calculated by HOSA toolbox

% N = size(Bispectrum,1);      % generally the same size as the object,
                               % even number in simulations
Beta = angle(Bispectrum);
Phi = zeros(1,N);

% Phi(1) = 0;

if mod(N,2)~=0
    Num = (N+1)/2;
    PhiHalf = zeros(1,Num);
else
    Num = N/2;
    PhiHalf = zeros(1,Num);
end

PhiHalf(1) = 0;

% 
% 
if mod(N,2)~=0
    iterNum = (N+1)/2;
else
    iterNum = N/2;
end

Sum_Temp(iterNum-1) = 0;

for r = 2:iterNum
    
    if mod(r,2)~=0
        const = (r-1)/2;
    else
        const = r/2;
    end
    
    for q = 1:floor(r/2)
        if mod(N,2)~=0
            Temp = PhiHalf(q)+PhiHalf(r-q)-Beta(-q+((N+1)/2),(r-q)+((N+1)/2));
            Disp1 = num2str(-q+((N+1)/2));
            Disp2 = num2str((r-q)+((N+1)/2));
            Dis = strcat('(',Disp1,',',Disp2,')');
%             disp(Dis)

        else
            Temp = PhiHalf(q)+PhiHalf(r-q)-Beta(-q+N/2+1,(r-q)+N/2+1);
            Disp1 = num2str(-q+N/2+1);
            Disp2 = num2str((r-q)+N/2+1);
            Dis = strcat('(',Disp1,',',Disp2,')');
%             disp(Dis);
        end 
        Exp_Temp = exp(1i*Temp);
        Sum_Temp(r-1) = Sum_Temp(r-1) + Exp_Temp;
    end
        Aver_Exp_Temp = (1/const)*Sum_Temp(r-1);
        PhiHalf(r) = angle(Aver_Exp_Temp);
        
        if mod(N,2)~=0
            Phi(1) = PhiHalf(1);
            Phi(Num) = PhiHalf(1);
            Phi(Num+r-1) = PhiHalf(r);
            Phi(Num-r+1) = -PhiHalf(r);
        else
            Phi(Num+1) = PhiHalf(1);
            Phi(Num+r) = PhiHalf(r);
            Phi(Num-r+2) = -PhiHalf(r);
        end
%         if mod(N,2)~=0
%             Phi(N+1-r) = -Phi(r);
%         else
%             Phi(N+2-r) = -Phi(r);
%         end
end
% Phi = ifftshift(Phi);
% Phi_Mod = exp(1i*Phi).*(ones(1,N)*exp(1i*pi));
% Phi = angle(Phi_Mod);

        
end

