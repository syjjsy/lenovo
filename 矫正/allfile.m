path='D:\APP\pylonpic\data\';
outpathl='D:\APP\pylonpic\all\1\';
outpath2='D:\APP\pylonpic\all\2\';
list=dir(path);
fileForm = '*.bmp';
ii=1;
for k=3:size(list,1)
    path1=strcat(path,list(k).name,'\');
    list1=dir(path1);
    path11=strcat(path1,'1','\');
    path22=strcat(path1,'2','\');
    files1 = dir(fullfile(path11,fileForm)); 
    len1 = size(files1,1);
    a=[ ];
    for i=1:len1
        numa=str2num(files1(i).name(1:end-4));
        a=[a;numa];
        
    end
    a=sort(a);
    for i=1:len1
        filename1 = strcat(path11,files1(i).name);
        filename2 = strcat(path22,files1(i).name);
        initialImage1 = imread(filename1);
        initialImage2 = imread(filename2);
        numb=str2num(files1(i).name(1:end-4));
        numbb=find(a==numb);
        outfile=sprintf('%04d',ii-i+numbb);
        outfile=strcat(outfile,'.bmp');
        finalPath1 = strcat(outpathl,outfile);
        imwrite(initialImage1,finalPath1);
        finalPath2 = strcat(outpath2,outfile);
        imwrite(initialImage2,finalPath2);
        ii=ii+1;
    end
    
end