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
% This code compares the gallery and probe iris binary codes and 
% compensates for the eye rotation by shifting the probe code +/- MAX_SHIFT 
% pixels. The comparison score is calculated according to the equation (3) 
% in the paper.

function scoreC = pmIrisBSIFMatch(codeBinary1,codeBinary2,msk1,msk2,filtersize)

%% Remove upper and lower boundaries depending on the filter size
%  But keep left/right portions due to circular symmetry of the normalized 
%  iris image.
margin = ceil(filtersize/2);
codeBinary1 = codeBinary1(margin:end-margin,:,:);
codeBinary2 = codeBinary2(margin:end-margin,:,:);
msk1 = msk1(margin:end-margin,:);
msk2 = msk2(margin:end-margin,:);

%% Use elements that were not masked in both images 
msk = msk1.*msk2;
[~,~,BITS] = size(codeBinary1);

%% Calculate the iris codes for each kernel in the filter set independently
MAX_SHIFT = 16;
scoreC = zeros(BITS,2*MAX_SHIFT+1);
for shift = -MAX_SHIFT:MAX_SHIFT
    for b=1:BITS    
        xorCodes = xor(squeeze(codeBinary1(:,:,b)),circshift(squeeze(codeBinary2(:,:,b)),shift,2));
        andMasks = msk1 & circshift(msk2,shift,2);
        xorCodesMasked = xorCodes & andMasks;
        scoreC(b,shift + MAX_SHIFT + 1) = sum(xorCodesMasked) / sum(andMasks);
    end
end

%% Finally, calculate the mean score for all the scores obtained for each 
%  kernel independently, and select the minimum value over all shifts
%  compensating the eye rotation
scoreC = min(mean(scoreC));
