function [pupilData, irisData, status] = pmIrisCircApprox(MASK)

% Make sure it's a logical array
MASK = logical(MASK);

if bwarea(MASK) < 64
    
    status = 'mask too small';
    
else
    
    status = [];
    
    % Some constants:
    lower_margin_pupil = 10;
    upper_margin_pupil = 10;
    lower_margin_iris = 5;
    upper_margin_iris = 20;
    
    polar_width = 512;
    
    % Extracting the largest blob (safeguard against DCNN fails)
    MASK = bwareafilt(MASK, 1);
    
    % Crop the image to the blob to estimate the iris size
    MASK_CROPPED = cropmat(MASK, MASK);
    
    IRIS_RADIUS_ESTIMATE = round(max(size(MASK_CROPPED))/2); % use larger of the sizes
    
    % Create the iris radius search range for Hough transform
    SEARCH_RANGE_I = [max(1,(IRIS_RADIUS_ESTIMATE - lower_margin_iris)) (IRIS_RADIUS_ESTIMATE + upper_margin_iris)];
    
    % Estimate the outer iris circle with Hough
    [IRIS_CENTERS, IRIS_RADII, ~] = imfindcircles(MASK, SEARCH_RANGE_I, 'ObjectPolarity', 'bright', 'Sensitivity', 0.99);
    
    % If no circles found do a second iteration with the smaller radius:
    if isempty(IRIS_CENTERS)
        
        IRIS_RADIUS_ESTIMATE = round(min(size(MASK_CROPPED))/2);
        
        % Create the iris radius search range for Hough transform
        SEARCH_RANGE_I = [(IRIS_RADIUS_ESTIMATE - lower_margin_iris) (IRIS_RADIUS_ESTIMATE + upper_margin_iris)];
        % estimate the outer iris circle with Hough
        
        [IRIS_CENTERS, IRIS_RADII, ~] = imfindcircles(MASK, SEARCH_RANGE_I, 'ObjectPolarity', 'bright', 'Sensitivity', 0.99);
        
    end
    
    if isempty(IRIS_CENTERS)
        
        status = 'iris approximation failed';
        
    else
        
        irisXY = IRIS_CENTERS(1,:);
        irisData.x = irisXY(1);
        irisData.y = irisXY(2);
        irisData.r = IRIS_RADII(1);
        
        %% CENTROID-BASED HEURISTIC for choosing iris radius range
        
        MASK_CROPPED_INVERSE = logical(1 - MASK_CROPPED);
        
        % Crop 5% from each side of the image to close most openings
        CROP = max(1,round(0.05 * max(size(MASK_CROPPED_INVERSE))));
        MASK_CROPPED_INVERSE = MASK_CROPPED_INVERSE(CROP:end-CROP, CROP:end-CROP);
        
        % Locate centroids
        CENTROID_STATS = regionprops('table',MASK_CROPPED_INVERSE,'Centroid', 'MajorAxisLength','MinorAxisLength');
        CENTROID_CENTERS = CENTROID_STATS.Centroid;
        CENTROID_RADII = round(mean([CENTROID_STATS.MajorAxisLength CENTROID_STATS.MinorAxisLength],2)/2);
        
        % Choose the one closest to the center of the image
        IMAGE_CENTER = flip(round(size(MASK_CROPPED_INVERSE)/2));
        CENTROID_DISTANCE_FROM_CENTER = pdist2(CENTROID_CENTERS, IMAGE_CENTER);
        PUPIL_RADIUS_ESTIMATE = round(CENTROID_RADII(CENTROID_DISTANCE_FROM_CENTER == min(CENTROID_DISTANCE_FROM_CENTER)));
        
        % In cases when it's difficult to find the pupil
        if (PUPIL_RADIUS_ESTIMATE - lower_margin_pupil) <= 0
            
            pupilData.x = irisData.x;
            pupilData.y = irisData.y;
            pupilData.r = round(irisData.r/3);
            
        else
            
            % Do the usual search routine
            SEARCH_RANGE_P = [(PUPIL_RADIUS_ESTIMATE - lower_margin_pupil) (PUPIL_RADIUS_ESTIMATE + upper_margin_pupil)];
            
            % Estimate the inner iris circle with Hough
            [PUPIL_CENTERS, PUPIL_RADII, ~] = imfindcircles(MASK, SEARCH_RANGE_P, 'ObjectPolarity', 'dark', 'Sensitivity', 0.99);
            
            % debug
            % viscircles(IRIS_CENTERS(1,:), IRIS_RADII(1,:),'Color','b'); % strongest candidate used
            % viscircles(PUPIL_CENTERS(1,:), PUPIL_RADII(1,:),'Color','r'); % strongest candidate used
            
            % If no circles found do a second iteration with the smaller radius:
            if isempty(PUPIL_CENTERS)
                
                status = 'pupil approximation failed';
                
            else
                
                pupilXY = PUPIL_CENTERS(1, :);
                pupilData.x = pupilXY(1);
                pupilData.y = pupilXY(2);
                pupilData.r = PUPIL_RADII(1);
                
            end
            
        end
        
    end
    
end

%% Generate points around the pupil and iris circles (e.g., for OSIRIS)
if isempty(status)
    pupilData.theta = 2*pi*(0:1/polar_width:1-1/polar_width);
    pupilData.xCirclePoints = round(pupilData.x + pupilData.r*cos(pupilData.theta));
    pupilData.yCirclePoints = round(pupilData.y + pupilData.r*sin(pupilData.theta));
    
    irisData.theta = pupilData.theta;
    irisData.xCirclePoints = round(irisData.x + irisData.r*cos(irisData.theta));
    irisData.yCirclePoints = round(irisData.y + irisData.r*sin(irisData.theta));
    
    status = 'OK';
else
    % something (see 'status') went wrong:
    pupilData = [];
    irisData = [];
end

