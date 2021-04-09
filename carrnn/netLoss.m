function [loss, NetParam] = netLoss(NetParam, y)

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

% Calculating the network loss and output gradients

% NetParam: structure including the network and optimization parameters, memory blocks, and gradients
% y: target array [outputSize x sequenceSize] for regression and [1 x sequenceSize] for classification
% loss: network loss

% Handling missing values in the target array based on the number of feature nodes per timestamp
outputFactor = (NetParam.outputSize - sum(isnan(y), 1)) / NetParam.outputSize; % output normalization factor
denomFactor = repmat(outputFactor, NetParam.outputSize, 1); % denominator factor
NetParam.outputActive(isnan(y)) = 0; % output values associated with missing target values are replaced with zeros
y(isnan(y)) = 0; % missing target values are replaced with zeros

% Average loss (cost) function associated with the number of available feature nodes per timestamp and the number of available timestamps
loss = sum(sum((NetParam.outputActive - y) .^ 2 ./ (NetParam.sequenceSize * NetParam.outputSize * denomFactor)));

% Gradients of the normalized loss w.r.t. network output with zero missing error propagation
NetParam.deltaOutputActive = (NetParam.outputActive - y) ./ (2 * NetParam.sequenceSize * NetParam.outputSize * denomFactor);

end
