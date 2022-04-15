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

% Custom data splitting layer

classdef splitLayer < nnet.layer.Layer

    properties
        FirstIndex
        LastIndex
    end

    methods

        function layer = splitLayer(firstIndex, lastIndex, name)
            layer.Name = name;
            layer.Description = 'Data splitter';
            layer.FirstIndex = firstIndex;
            layer.LastIndex = lastIndex;
        end

        function Z = predict(layer, X)
            Z = X(layer.FirstIndex : layer.LastIndex, :); % output [features x batch]
            numFeatures = layer.LastIndex - layer.FirstIndex + 1; % number of features
            K = (numFeatures - sum(~(Z > - realmax & Z < realmax), 1)) / numFeatures; % normalized number of features [1 x batch]
            Z = Z .* K; % normalized with the number of available features
            Z(~(Z > - realmax & Z < realmax)) = 0; % missing values replaced with zeros
        end

    end

end