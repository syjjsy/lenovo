lujing='Q:\0825高灵敏新数据85add\';
% path_save='Q:\syj\825\mat\';
numm=1;
for i=1001:2000;
    pa=strcat(lujing,sprintf('%04d',i),'\0001\');
    nma=dir([pa,'*.bmp']);
    iiss=[];
    if(length(nma)~=1024)
       disp(pa);disp(numm);
       numm=numm+1;
       continue;
    end
    for ii=1:length(nma)
        I=imread([pa,nma(ii).name]);
        iii=I(324:343,628:646);
        iis=sum(sum(iii));
        
        iiss=[iiss;iis];
    end
%     length(iiss)
    path_save=strcat(path_save,num2str(i),'.mat');
    save (['Q:\syj\825\mat\',num2str(i)],'iiss');
end
