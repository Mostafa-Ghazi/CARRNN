function df = activationDerivativeIn(x, type)

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
%%%%%                https://arxiv.org/abs/                           %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Derivative of the activation function using its input array

% x: activation input [numNodes x numObservation]
% df: activation derivative using its input [numNodes x numObservation]
% type: name of the activation function

switch type
    case 'identity'
        df = 1;
    case 'sigmoid'
        df = exp(- x) ./ (1 + exp(- x)) .^ 2;
    case 'tanh'
        df = 4 * exp(- 2 * x) ./ (1 + exp(- 2 * x)) .^ 2; % df = 1 - tanh(x) .^ 2;
    case 'efftanh'
        df = 1.7159 * 2 / 3 * (1 - tanh(2 / 3 * x) .^ 2);
    case 'softplus'
        df = 1 ./ (1 + exp(- x));
    case 'relu'
        df = double(x >= 0);
    case 'prelu'
        alpha = 0.01;
        df = double(x >= 0) + alpha * double(x < 0);
    case 'selu'
        alpha = 1.67326;
        lambda = 1.0507;
        df = lambda * (double(x >= 0) + alpha * exp(x) .* (x < 0));
    case 'softmax'
        x = exp(bsxfun(@minus, x, max(x, [], 1))); % subtracted from max to avoid NaNs
        f = bsxfun(@rdivide, x, sum(x, 1)); % softmax activation output
        df = f .* (1 - f); % diagonal derivatives
end

end