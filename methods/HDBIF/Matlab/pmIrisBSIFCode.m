% Supplementary material for the paper:
% Adam Czajka, Daniel Moreira, Kevin W. Bowyer, Patrick J. Flynn, 
% "Domain-Specific Human-Inspired Binarized Statistical Image Features for 
% Iris Recognition," WACV 2019, Hawaii, 2019
% 
% Pre-print available at: https://arxiv.org/abs/1807.05248
% 
% Please follow the instructions at https://cvrl.nd.edu/projects/data/ 
% to get a copy of the test database.
%
% This code has been adapted for iris recognition purposes from the
% original BSIF Matlab code supplementing the paper by Kannala and Rahtu, 
% "BSIF: binarized statistical image features", ICPR 2012, and available at
% http://www.ee.oulu.fi/~jkannala/bsif/bsif.html

function codeBinary = pmIrisBSIFCode(img,texturefilters)

%% Initialization
img = double(img);
numScl = size(texturefilters,3);
codeBinary = zeros([size(img) numScl]);

%% Wrap image
r = floor(size(texturefilters,1)/2);
upimg = img(1:r,:);
btimg = img((end-r+1):end,:);
lfimg = img(:,1:r);
rtimg = img(:,(end-r+1):end);
cr11 = img(1:r,1:r);
cr12 = img(1:r,(end-r+1):end);
cr21 = img((end-r+1):end,1:r);
cr22 = img((end-r+1):end,(end-r+1):end);
imgWrap = [cr22,btimg,cr21;rtimg,img,lfimg;cr12,upimg,cr11];

%% Loop over all kernels in a given set and calculate iris binary codes
for i=1:numScl
  tmp = texturefilters(:,:,numScl-i+1);
  ci = filter2(tmp,imgWrap,'valid');
  codeBinary(:,:,i) = (ci>0);
end