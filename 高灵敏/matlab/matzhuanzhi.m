clear all;
clear all;

pathl='E:\glmdata\mat\';
fileForm = '*.mat';
files1 = dir(fullfile(pathl,fileForm)); 
len1 = size(files1,1);
for i=1:len1
    filename1 = strcat(pathl,files1(i).name);
    eval(['load ' filename1])
    iiss=iiss';
%     I=imread([pa,nma(ii).name]);
%     iii=I(324:343,628:646);
%     iis=sum(sum(initialImage1)); 
    
%     iiss=[iiss;iis];
    save (['E:\glmdata\mat\zz\',num2str(i)],'iiss');
end
