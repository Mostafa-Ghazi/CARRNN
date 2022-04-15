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

% Custom regression output layer with normalized loss

classdef outputRegressionLayer < nnet.layer.RegressionLayer

    methods

        function layer = outputRegressionLayer(name)
            layer.Name = name;
            layer.Description = 'Regression loss';
        end

        function L = forwardLoss(layer, Y, T)
            [K, I, J] = size(Y); % number of features, minibatch observations, and timestamps
            K = (K - sum(~(T > - realmax & T < realmax), 1)); % number of available features [1 x batch x times]
            L = (Y - T) .^ 2 ./ (I * J * K); % errors normalized with the number of available features
            L(~(L > - realmax & L < realmax)) = 0; % missing values replaced with zeros
            L = sum(L(:)); % mean squared error loss
        end

        function dLdY = backwardLoss(layer, Y, T)
            [K, I, J] = size(Y); % number of features, minibatch observations, and timestamps
            K = (K - sum(~(T > - realmax & T < realmax), 1)); % number of available features [1 x batch x times]
            dLdY = 2 * (Y - T) ./ (I * J * K); % errors normalized with the number of available features
            dLdY(~(dLdY > - realmax & dLdY < realmax)) = 0; % missing values replaced with zeros
        end

    end

end