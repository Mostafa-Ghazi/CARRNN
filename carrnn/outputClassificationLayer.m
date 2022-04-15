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

% Custom classification output layer with normalized loss

classdef outputClassificationLayer < nnet.layer.ClassificationLayer

    methods

        function layer = outputClassificationLayer(name)
            layer.Name = name;
            layer.Description = 'Classification loss';
        end

        function L = forwardLoss(layer, Y, T)
            [K, I, J] = size(Y); % number of classes, minibatch observations, and timestamps
            J = (J - sum(~(T > - realmax & T < realmax), 3)); % number of available timestamps [K x batch x 1]
            T(~(T > - realmax & T < realmax)) = 0; % missing values replaced with zeros
            L = - T .* log(Y + 1e-8) ./ (I * J); % cross-entropy errors normalized with the number of available timestamps
            L = sum(L(:)); % accumulated loss
        end

    end

end