clear all;
pathl='E:\glmdata\glmatlab\±àÂë\';
fileForm = '*.bmp';
files1 = dir(fullfile(pathl,fileForm)); 
len1 = size(files1,1);
iiss=[];
for i=1:len1
    filename1 = strcat(pathl,files1(i).name);
    initialImage1 = imread(filename1);
%     I=imread([pa,nma(ii).name]);
%     iii=I(324:343,628:646);
    iis=sum(sum(initialImage1)); 
    
    iiss=[iiss;iis];
    
end
save (['E:\glmdata\glmatlab\',num2str(i)],'iiss');