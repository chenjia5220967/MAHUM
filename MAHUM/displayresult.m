clc
clear all
close all
currentFolder = pwd;
addpath(genpath(currentFolder));
addpath(genpath('src'));
addpath(genpath('result'));
addpath(genpath('result\jasper'));
addpath(genpath('result\samson'));

%% load Moffett Data
  %    load Y_Moffett.mat
 %    load Y3d.mat
  %   load jasper_dataset.mat
%     load samson_dataset.mat

%    GT=double(A');
%    [bands,n] = size(Y);   
%   style = 1; % VCA is used for initialization
%    style = 2; % N-FINDR is used for initialization
%    maxiteration = 100;
    p = 4;
    col=100;

    
    Files = dir(fullfile('D:\坚果云\我的坚果云\我的坚果云\代码\Yuanchao_Unmixing_Codes\DMBU\demo_DMBU\result\jasper\5\','*.mat'));

LengthFiles = length(Files);

for i = 1:LengthFiles;
        load (Files(i).name);
        if i==1
            Abundance11 =A;
            BAbundance11 = B;
        else
            Abundance11 =[Abundance11;A];
            BAbundance11 = [BAbundance11;B];
        end
end
    
%     load 460jasper22-Nov-2022-13-10-13.mat
%        Abundance1 =A;
%        BAbundance1 = B;
%     load 480jasper22-Nov-2022-13-18-37.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%     load 480jasper22-Nov-2022-13-24-18.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B]; 
%      load 480jasper22-Nov-2022-13-29-16.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B]; 
%      load 480jasper22-Nov-2022-13-33-51.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];  
%     load 480jasper22-Nov-2022-13-37-26.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%      load 480jasper22-Nov-2022-13-40-18.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%     load 480jasper22-Nov-2022-13-42-37.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%     load 480jasper22-Nov-2022-13-44-05.mat
%     Abundance11 =[Abundance1;A];
%        BAbundance11 = [BAbundance1;B];
%      load D:\坚果云\我的坚果云\我的坚果云\代码\Yuanchao_Unmixing_Codes\DMBU\demo_DMBU\result\samson\ground\samson0.5_cycunet_result.mat
%      A=reshape(double(A),3,col*col);
    
    

    
    
   m=p*(p-1)/2; 

  Abundance11=Abundance11';
  BAbundance11=BAbundance11';
   
    figure
    for i=1:p*LengthFiles
         subplot(LengthFiles,p,i)
          map = reshape(Abundance11(:,i),[col,col]);
         imagesc(map);axis off;
    end
    
%     gt=reshape(A',col,col,p);
%          subplot(10,p,p*(10-1)+1)
%         imagesc(gt(:,:,2)');axis off
%                 subplot(10,p,p*(10-1)+2)
%         imagesc(gt(:,:,3)');axis off
%                 subplot(10,p,p*(10-1)+3)
%         imagesc(gt(:,:,1)');axis off
%     colormap(jet);

    
    
    
 
     figure
    for i=1:m*LengthFiles
        subplot(LengthFiles,m,i)
         map = reshape(BAbundance11(:,i),[col,col]);
        imagesc(map);
        title(i);
        axis off;
    end
            colormap(jet);
  
            
            
            
            
            
%      load 410jasper22-Nov-2022-15-59-26.mat
%     Abundance1 =A;
%        BAbundance1 = B;
%     load 420jasper22-Nov-2022-15-58-15.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%     load 430jasper22-Nov-2022-15-56-28.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B]; 
%      load 440jasper22-Nov-2022-15-53-52.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B]; 
%      load 450jasper22-Nov-2022-15-50-54.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];  
%     load 460jasper22-Nov-2022-15-47-19.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%      load 470jasper22-Nov-2022-15-43-15.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%     load 480jasper22-Nov-2022-15-38-37.mat
%     Abundance1 =[Abundance1;A];
%        BAbundance1 = [BAbundance1;B];
%     load 480jasper22-Nov-2022-13-44-05.mat
%     Abundance12 =[Abundance1;A];
%        BAbundance12 = [BAbundance1;B];
%     load jasper0.5_cycunet_result.mat
%     A=reshape(double(A),p,col*col);               
%   Abundance12=Abundance12';
%   BAbundance12=BAbundance12';
%    
% %     figure
% %     for i=1:p*9
% %          subplot(10,p,i)
% %           map = reshape(Abundance12(:,i),[col,col]);
% %          imagesc(map);axis off;
% %     end
% %     
% %     gt=reshape(A',col,col,p);
% %          subplot(10,p,p*(10-1)+1)
% %         imagesc(gt(:,:,1)');axis off
% %                 subplot(10,p,p*(10-1)+2)
% %         imagesc(gt(:,:,3)');axis off
% %                 subplot(10,p,p*(10-1)+3)
% %         imagesc(gt(:,:,2)');axis off
% %         subplot(10,p,p*(10-1)+4)
% %         imagesc(gt(:,:,4)');axis off
% % 
% %     colormap(jet);
% % 
% %     
% %     
% %     
% %  
% %      figure
% %     for i=1:m*9
% %         subplot(9,m,i)
% %          map = reshape(BAbundance12(:,i),[col,col]);
% %         imagesc(map);
% %         title(i);
% %         axis off;
% %     end
% %             colormap(jet);
%             
%             
% abu_est=reshape(double(abu_est),p,col*col);            
% abu_cube=reshape(abu_est',col,col,p);
% 
% 
%   
%  image1= reshape(Abundance11(:,20),[col,col]);       
%    image2= reshape(BAbundance11(:,6),[col,col]);    
%     image3= reshape(BAbundance11(:,30),[col,col]); 
%     image4=image1+500*image2;
%     image5=image1+500*image2+image3;
%             
%      figure
%       subplot(1,5,1)
%      map =  image1;
%         imagesc(map);axis off;
%               subplot(1,5,2)
%    map =  image2;
%                      imagesc(map);axis off;
%               subplot(1,5,3)
%     map =  image3;
%                      imagesc(map);axis off;
%                                    subplot(1,5,4)
%     map = image4;
%                      imagesc(map);axis off;
%                                     subplot(1,5,5)
%     map = image5;
%                      imagesc(map);axis off;           
% %  52 20 26
% %   19 21 
% %  
% %  53 54 24
%  
%  
% % %% GT
% % if display_if
% %     gt=reshape(GT,col,col,p);
% %     for i=1:p
% %         subplot(2,p,5-i+p)
% %         imagesc(gt(:,:,i));axis off
% %     end
% %     colormap(jet);
% % end
% % 
% % end    
% % m=6
% % Babundance = B';
% %   if display_if
% %     figure
% % 
% %     for i=1:m
% %         subplot(2,4,i)
% %          map = reshape(Babundance(:,i),[100,100]);
% %         imagesc(map);axis off;
% %     end
% %     colormap(jet);
% %   end     
% % 
% %   l=3
% %   
% %     figure
% % % if display_if
% % %     for i=1:p
% % %         subplot(3,p,i)
% % %          map = reshape(Abundance(:,i),[col,col]);
% % %         imagesc(map);axis off;
% % %     end
% % %     colormap(jet);
% % % end       
% %        subplot(l,4,1)
% %               map = reshape(Abundance(:,4),[col,col]);
% %                      imagesc(map);axis off;    
% %        subplot(l,4,2)
% %               map = reshape(Abundance(:,2),[col,col]);
% %                      imagesc(map);axis off;    
% %        subplot(l,4,3)
% %               map = reshape(Abundance(:,1),[col,col]);
% %                      imagesc(map);axis off;    
% %        subplot(l,4,4)
% %             map = reshape(Abundance(:,3),[col,col]);
% %                       imagesc(map);axis off;        
% %        
% %        subplot(l,4,5)
% %               map = reshape(Babundance(:,5),[100,100]);
% %                      imagesc(map);axis off;    
% %        subplot(l,4,6)
% %               map = reshape(Babundance(:,2),[100,100]);
% %                      imagesc(map);axis off;    
% %        subplot(l,4,7)
% %               map = reshape(Babundance(:,1),[100,100]);
% %                      imagesc(map);axis off;    
% %        subplot(l,4,8)
% %        map = reshape(Babundance(:,6),[100,100]);
% %               imagesc(map);axis off;   
% %               
% %               
% % %        subplot(4,4,9)
% % %        B1=Babundance(:,5)+Abundance(:,4);
% % %               map = reshape(Babundance(:,5)+Abundance(:,4),[100,100]);
% % %                      imagesc(map);axis off;    
% % %        subplot(4,4,10)
% % %        B2=Babundance(:,2)+Abundance(:,2);
% % %               map = reshape(Babundance(:,2)+Abundance(:,2),[100,100]);
% % %                      imagesc(map);axis off;    
% % %        subplot(4,4,11)
% % %         B3=Babundance(:,1)+Abundance(:,1);
% % %               map = reshape(B3,[100,100]);
% % %                      imagesc(map);axis off;    
% % %        subplot(4,4,12)
% % %        B4=Abundance(:,3)+Babundance(:,6);
% % %        map = reshape( B4,[100,100]);
% % %               imagesc(map);axis off;   
% %               
% %      colormap(jet);      
% %      
% %  if display_if
% %     gt=reshape(GT,col,col,p);
% %     for i=1:p
% %         subplot(l,4,5-i+4*(l-1))
% %         imagesc(gt(:,:,i));axis off
% %     end
% %     colormap(jet);
% % end      
% %        
% % M_est=EndmemberEst(Y,A,300); % estimate endmembers by ||X-MA||, when A is given.
% % %% Evalution metrics
% % rmse=sqrt(sum(sum((GT-A').^2))/(p*col*col))
% % [SAD,SADerr] = SadEval(M_est,E)
% 
%     
    