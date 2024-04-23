% pmIrisSegment Iris image segmentation.
%
%    mask = pmIrisSegment(image,net) calculates binary mask
%    indicating iris (white mask elements) and non-iris (black mask pixels)
%    areas for an input iris image 'image', using the SegNet model 'net'.
%
%    Input:
%    image     - iris image, preferably compliant with ISO/IEC 19794-6
%    netCoarse - trained segmentation model (MATLAB DAGNetwork) used in
%                circular boundaries estimation 
% 
%    Output:
%    mask      - 0/1 binary mask (size same as input image)

function mask = pmIrisSegment(image,net)

    % Current version of SegNet was trained for 320x240 resolution
    imageSize = size(image);
    
    if length(imageSize) > 2
        image = rgb2gray(image);
        imageSize = imageSize(1:2);
    end
    
    if (imageSize(1) ~= 240 && imageSize(2) ~= 320)
        image = imresize(uint8(image), [240 320], 'bicubic');
    end
    
    % Run the semantic segmentation network
    prediction_categorical = semanticseg(image, net);
   
    % Convert the categorical output to ISO/IEC mask
    background = zeros(size(prediction_categorical));
    overlay = labeloverlay(background, prediction_categorical);
    overlay = overlay(:, :, 2);
    
    % Get pixel IDs for iris and background
    irisIDX = overlay == 0;
    backgroundIDX = overlay == 128;
    
    % Switch the values
    overlay(irisIDX) = 255;
    overlay(backgroundIDX) = 0;
    mask = overlay;
    mask = imbinarize(imresize(mask, imageSize, 'nearest'));
   
end