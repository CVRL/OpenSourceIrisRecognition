% pmSegNetAnnotate Visualization of iris image segmentation.
%
%    annotatedImage = pmSegNetAnnotate(image,mask,pupilData,irisData)
%    annotates the image with binary 'mask' and circular approximations in
%    'pupilData' and 'irisData' structures.
% 
%    Input:
%    image               - iris image
%    mask                - 0/1 binary mask (of the same size as the input image)
%    pupilData           - x, y and r of the circle aproximating the pupil
%    irisData            - x, y and r of the circle aproximating the iris
% 
%    Output:
%    annotatedImage      - 640x480 color image with segmentation result

function annotatedImage = pmSegNetAnnotate(image,mask,pupilData,irisData)

[~,~,ch] = size(image);

if ch < 3
    image = cat(3, image, image, image);
elseif ch > 3
    image = image(:,:,1:3);
end

image = double(imresize(image,[480 640],'bicubic'));

annotatedImage = double(image);
annotatedImage(:,:,1) = min(255,annotatedImage(:,:,1) + 0.4*255*double(mask));
annotatedImage(:,:,2) = min(255,annotatedImage(:,:,2) + 0.4*255*double(mask));

annotatedImage = uint8(annotatedImage);

if ~isempty(pupilData)
    annotatedImage = insertShape(annotatedImage, 'Circle', [pupilData.x pupilData.y pupilData.r], 'LineWidth', 3, 'Color', 'red');
end

if ~isempty(irisData)
    annotatedImage = insertShape(annotatedImage, 'Circle', [irisData.x irisData.y irisData.r], 'LineWidth', 3, 'Color', 'blue');
end
