function [y, outputStamps, visitInterval] = dataBinning(x, inputStamps, tau)

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

% Time binning and alignment of the input array events

% x: input array [numFeatures x inputSequenceSize]
% inputStamps: input timestamps array [1 x inputSequenceSize]
% tau: regular time step (uniform bin width)
% objective: network objective (regression or classification)
% y: aligned input array [numFeatures x outputSequenceSize]
% outputStamps: output timestamps array [1 x inputSequenceSize]
% visitInterval: time intervals of the aligned visits [1 x outputSequenceSize]

% Aligning the feature visits
outputStamps = min(inputStamps) : tau : max(inputStamps); % timestamps with regular steps
y = NaN(size(x, 1), length(outputStamps)); % aligned features
for n = 1 : size(y, 1) % loop over the number of features
    featureIDX = find(~isnan(x(n, :))); % indices of the available time points for the selected feature
    if numel(featureIDX)
        [~, minIDX] = min(abs(bsxfun(@minus, outputStamps, inputStamps(featureIDX)')), [], 2); % indices of the closest regular points to the quarry
        % minIDX = dsearchn(outputStamps', inputStamps(featureIDX)'); % indices of the closest regular points to the quarry
        y(n, minIDX) = x(n, featureIDX); % matched feature points
    end
end

% Removing the empty data columns (timestamps)
emptyIDX = find(isnan(y(end, :)) | sum(~isnan(y(1 : end - 1, :)), 1) < ceil(0.05 * (size(y, 1) - 1))); % indices of the (nearly) empty data and label columns
outputStamps(emptyIDX) = []; % empty columns removed from regularly-spaced timestamps
y(:, emptyIDX) = []; % empty columns removed from data
visitInterval = diff(outputStamps, 1, 2); % time intervals of the aligned visits

end
