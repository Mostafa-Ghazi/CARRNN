%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                 Source Code Author Information                  %%%%%
%%%%%                     Mostafa Mehdipour Ghazi                     %%%%%
%%%%%                   mostafa.mehdipour@gmail.com                   %%%%%
%%%%%                      Created on 01/04/2021                      %%%%%
%%%%%                      Updated on 01/04/2022                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  This code is an implementation of the algorithm published in:  %%%%%
%%%%%  CARRNN: A Continuous Autoregressive Recurrent Neural Network   %%%%%
%%%%%  for Deep Representation Learning from Sporadic Temporal Data   %%%%%
%%%%%                https://arxiv.org/abs/2104.03739                 %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Custom autoregression layer with exponential weights

classdef carLayer < nnet.layer.Layer

    properties
        NumFeatures
        Tau
    end

    properties (Learnable)
        Weights
        Bias
    end

    methods

        function layer = carLayer(numFeatures, tau, name)
            layer.Name = name;
            layer.Tau = tau;
            layer.Description = 'Continuous autoregression';
            layer.NumFeatures = numFeatures; % number of features
            layer.Bias = zeros(layer.NumFeatures, 1); % bias initialization [features x 1]
            if numel(layer.Tau)
                layer.Weights = zeros(layer.NumFeatures, layer.NumFeatures); % weights initialization [featuresOut x featuresIn]
            else
                layer.Weights = zeros(layer.NumFeatures, 1); % weights initialization [features x 1]
            end
        end

        function Z = predict(layer, X)
            Z = X(1 : layer.NumFeatures, :); % input data [features x batch]
            if numel(layer.Tau)
                dt = X(end, :) - layer.Tau; % time interval [1 x batch]
                Z = Z + (layer.Weights * Z) .* dt + layer.Bias * dt; % irregular output [features x batch]
            else
                dt = X(layer.NumFeatures + 1 : 2 * layer.NumFeatures, :); % time lag [features x batch]
                Z = Z + layer.Weights .* Z .* dt + layer.Bias .* dt; % filled output [features x batch]
            end
        end

    end

end