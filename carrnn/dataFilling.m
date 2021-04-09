function [y, featureInterval] = dataFilling(x, stamps, fillingType)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                 Source Code Author Information                  %%%%%
%%%%%                     Mostafa Mehdipour Ghazi                     %%%%%
%%%%%                   mostafa.mehdipour@gmail.com                   %%%%%
%%%%%                      Created on 01/04/2018                      %%%%%
%%%%%                      Updated on 01/04/2021                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  This code is an implementation of the algorithm published in:  %%%%%
%%%%%  CARRNN: A Continuous Autoregressive Recurrent Neural Network   %%%%%
%%%%%  for Deep Representation Learning from Sporadic Temporal Data   %%%%%
%%%%%                https://arxiv.org/abs/2104.03739                 %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Missing data imputation of the aligned events

% x: aligned input array [numFeatures x sequenceSize]
% stamps: aligned timestamps array [1 x sequenceSize]
% fillingType: missing values imputation method
% y: filled input array [numFeatures x sequenceSize]
% featureInterval: time intervals of the features [numFeatures x sequenceSize]

% Filling the missing values with their neighboring points
y = x; % filled input array
featureInterval = zeros(size(y)); % time intervals of the features
switch fillingType
    case 'zero-filling'
        y(isnan(y)) = 0; % replace missing values with zeros
    case 'forward-filling'
        for n = 1 : size(y, 1) % loop over the number of features
            firstIDX = find(~isnan(y(n, :)), 1, 'first'); % index of the first available data feature point
            if numel(firstIDX)
                missingIDX = find(isnan(y(n, :))); % indices of the missing time points for the selected feature
                missingIDX = missingIDX(missingIDX > firstIDX); % indices of missing values beyond the first non-missing
                for t = missingIDX
                    y(n, t) = y(n, t - 1); % replace each missing value with the previous non-missing value
                    featureInterval(n, t) = stamps(t) - stamps(find(~featureInterval(n, 1 : t - 1), 1, 'last')); % time intervals
                end
            end
        end
        y(isnan(y)) = 0; % replace remaining missing values with zeros
    case 'nearest-filling'
        for n = 1 : size(y, 1) % loop over the number of features
            featureIDX = find(~isnan(y(n, :))); % indices of the available time points for the selected feature
            if numel(featureIDX)
                [~, minIDX] = min(abs(bsxfun(@minus, stamps, stamps(featureIDX)')), [], 1); % indices of the nearest points to the quarry
                y(n, :) = y(n, featureIDX(minIDX)); % data imputation using the neighboring points
                featureInterval(n, :) = stamps - stamps(featureIDX(minIDX)); % time intervals
            end
        end
        y(isnan(y)) = 0; % replace remaining missing values with zeros
end

end
