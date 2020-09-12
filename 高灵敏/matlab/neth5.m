lujing='Q:\syj\mat\';
path_save='Q:\syj\1000h5\';

na=dir(strcat(lujing,'*.mat'));
% % % % % 1*1024
for num=1:length(na)
   load(strcat(lujing,na(num).name))
   I=iiss;
   my_hight=size(I,1);
   my_width=size(I,2);
    
   ming=na(num).name(1:end-4);
   ming=str2num(ming);
   ming=ming+2000;
   ming=sprintf('%04d',ming);
   ming=num2str(ming);
   p3=strcat(path_save,ming,'.h5');
   dn=strcat('/',ming);
   if my_hight==1024
       h5create(p3,dn,[my_hight,my_width]);
   h5write(p3,dn,I);
   end
   

end