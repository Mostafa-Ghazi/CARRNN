function f = activationFunction(x, type)

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

% Network layer activation function

% x: activation input [numNodes x numObservation]
% f: activation output [numNodes x numObservation]
% type: name of the activation function

switch type
    case 'identity'
        f = x;
    case 'sigmoid'
        f = 1 ./ (1 + exp(- x)); % logistic or soft step
    case 'tanh'
        f = 2 ./ (1 + exp(- 2 * x)) - 1; % f = tanh(x);
    case 'efftanh'
        f = 1.7159 * tanh(2 / 3 * x); % (computationally) efficient tanh
    case 'softplus'
        f = log(1 + exp(x));
    case 'relu'
        f = x .* (x >= 0);
    case 'prelu'
        alpha = 0.01; % parametric or leaky
        f = x .* (x >= 0) + alpha * x .* (x < 0);
    case 'selu'
        alpha = 1.67326;
        lambda = 1.0507;
        f = lambda * (x .* (x >= 0) + alpha * (exp(x) - 1) .* (x < 0));
    case 'softmax'
        x = exp(bsxfun(@minus, x, max(x, [], 1))); % subtracted from max to avoid NaNs
        f = bsxfun(@rdivide, x, sum(x, 1));
end

end
