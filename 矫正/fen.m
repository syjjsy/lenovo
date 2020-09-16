clear all
path='D:\APP\pylonpic\all\2\';
outpath1='D:\APP\pylonpic\fen\2\wu\';
outpath2='D:\APP\pylonpic\fen\2\you\';
fileForm = '*.bmp';
files1 = dir(fullfile(path,fileForm)); 
len1 = size(files1,1);
for i=1:len1
    filename1 = strcat(path,files1(i).name);
    initialImage1 = imread(filename1);
    b=str2num(filename1(end-7:end-4));
    
    if(mod(b,2)==0)
        outfile=sprintf('%04d',b-1);
        outfile=strcat(outfile,'.bmp');
        finalPath1 = strcat(outpath2,outfile);
        imwrite(initialImage1,finalPath1);
    else
        outfile=sprintf('%04d',b);
        outfile=strcat(outfile,'.bmp');
        finalPath2= strcat(outpath1,outfile);
        imwrite(initialImage1,finalPath2);
    end
end