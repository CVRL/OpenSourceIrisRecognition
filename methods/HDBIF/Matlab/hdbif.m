% This sample demonstrates how to combine selected efforts aimed at
% delivering effective iris recognition methods for diseased eyes
% and post-mortem samples (collected after death).
%
% The following codes and models were merged here into a complete iris
% recognition software package:
%
% a) segmentation and normalization:
%    Mateusz Trokielewicz, Adam Czajka, Piotr Maciejewicz, ?Post-mortem
%    iris recognition with deep learning-based image segmentation,? Image
%    and Vision Computing, Vol. 94 (103866), Feb. 2020, pp. 1-11;
%    pre-print: https://arxiv.org/abs/1901.01708
%
% b) human-driven BSIF-based iris pattern encoding:
%    Adam Czajka, Daniel Moreira, Kevin W. Bowyer, Patrick Flynn,
%    ?Domain-Specific Human-Inspired Binarized Statistical Image Features
%    for Iris Recognition,? The IEEE Winter Conference on Applications
%    of Computer Vision, Waikoloa Village, Hawaii, January 7-11, 2019;
%    pre-print: https://arxiv.org/abs/1807.05248
% _________________________________________________________________________
% (c) Adam Czajka, aczajka@nd.edu
%     First version: October 2020, last update: April 17, 2024
%     Tested up to MATLAB ver. 9.13.0.2105380 (R2022b) Update 2

clear all
close all

%% Housekeeping

% where various things live:
DIR_IMAGES_TO_PROCESS = './data/';
DIR_IMAGES_PROCESSED = './dataProcessed/';
DIR_BSIF_FILTERS = './filters_txt/';
DIR_TEMPLATES = './templates/';

% domain-specific BIF kernel bank (based on WACV 2019 experiments / paper):
FILTER_BANK_SELECTED = 'finetuned_bsif_eyetracker_data/';
l = 17;     % size of the filter
n = 5;      % number of kernels in a set
filter_path = [DIR_BSIF_FILTERS FILTER_BANK_SELECTED 'ICAtextureFilters_' num2str(l) 'x' num2str(l) '_' num2str(n) 'bit.txt'];
ICAtextureFilters = reshape(readmatrix(filter_path), l, l, n);
%filters = ['../filters/' FILTER_BANK_SELECTED 'ICAtextureFilters_' num2str(l) 'x' num2str(l) '_' num2str(n) 'bit.mat'];
%load(filters);

% image lists:
compList = readtable('imageList.txt','Delimiter',' ');
compListScores = 'imageListScores.txt';

% segmentation models:
disp('Loading segmentation models ...')
load('./models/SegNetCoarse-MTPHD.mat')
modelCoarse = net;
load('./models/SegNetFine-MTPHD.mat')
modelFine = net;
clear net


%% Process all unique files on the list: segmentation, normalization, encoding

disp(['Reading the matching pairs file ...'])
uniqueFiles = unique([compList.file1;compList.file2]);
uniqueFilesL = length(uniqueFiles);

disp(['Found ' num2str(uniqueFilesL) ' unique files; segmenting, normalizing and encoding ...'])
for i = 1:uniqueFilesL
    
    disp([num2str(i) '/' num2str(uniqueFilesL) ': ' uniqueFiles{i}]);
    
    % step 1: segmentation
    % (coarse for circular approximations; fine for actual occlusion mask)
    
    filePath = uniqueFiles{i};
    image = imread(fullfile(DIR_IMAGES_TO_PROCESS,filePath));
    maskCoarse = pmIrisSegment(image, modelCoarse);
    maskFine = pmIrisSegment(image, modelFine);
    
    % step 2: circular approximation
    [pupilData, irisData, status] = pmIrisCircApprox(maskCoarse);
    
    if strcmp(status,'OK')

        % step 3: normalization
        [imagePol,maskPol] = pmIrisCartesianToPolar(image,maskFine,pupilData,irisData);
        
        % optional: visualization of steps 1-3 
        imwrite(imagePol,[DIR_IMAGES_PROCESSED filePath(1:end-4) '_im_polar.png'],'png');
        imwrite(maskPol,[DIR_IMAGES_PROCESSED filePath(1:end-4) '_mask_polar.png'],'png');
        imwrite(maskCoarse,[DIR_IMAGES_PROCESSED filePath(1:end-4) '_seg_mask_coarse.png'],'png');
        imwrite(maskFine,[DIR_IMAGES_PROCESSED filePath(1:end-4) '_seg_fine_mask.png'],'png');
        annotatedImage = pmSegNetAnnotate(image,maskFine,pupilData,irisData);
        imwrite(annotatedImage,[DIR_IMAGES_PROCESSED filePath(1:end-4) '_seg_vis.png'],'png');
        
        % step 4: feature extraction
        codePol = pmIrisBSIFCode(imagePol,ICAtextureFilters);
        save([DIR_TEMPLATES filePath(1:end-4) '_tmpl.mat'],'maskPol','codePol');

        % optional: visualization of step 4
        [~,~,numFilters] = size(codePol);
        for f = 1:numFilters
            imwrite(squeeze(codePol(:,:,f)),[DIR_IMAGES_PROCESSED filePath(1:end-4) '_code_filter' num2str(f-1) '.png'],'png');
        end

    else
        imwrite(image,[DIR_IMAGES_PROCESSED filePath(1:end-4) '_seg_vis.png'],'png');
    end
    
end


%% Process the matching list

compListH = height(compList);
disp(['Matching ' num2str(compListH) ' pairs ...'])

f = fopen(compListScores,'w+');

for i=1:compListH
    
    f1 = [DIR_TEMPLATES compList.file1{i}(1:end-4) '_tmpl.mat'];
    f2 = [DIR_TEMPLATES compList.file2{i}(1:end-4) '_tmpl.mat'];
    
    if exist(f1,'file') && exist(f2,'file')
        
        % code / mask #1
        load(f1);
        code1 = codePol;
        mask1 = maskPol;
        
        % code / mask #2
        load(f2);
        code2 = codePol;
        mask2 = maskPol;
        
        score = pmIrisBSIFMatch(code1,code2,mask1,mask2,l);
        
        fprintf(f,[compList.file1{i} ' ' compList.file2{i} ' ' num2str(score) '\n']);
        disp([compList.file1{i} ' <> ' compList.file2{i} ' = ' num2str(score)])
        
    else
        fprintf(f,[compList.file1{i} ' ' compList.file2{i} ' -1.0\n']);
    end
    
end

fclose(f);
