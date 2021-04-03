function df = activationDerivativeOut(f, type)

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

% Derivative of the activation function using its output array

% f: activation output [numNodes x numObservation]
% df: activation derivative using its output [numNodes x numObservation]
% type: name of the activation function

switch type
    case 'identity'
        df = 1;
    case 'sigmoid'
        df = f .* (1 - f);
    case 'tanh'
        df = 1 - f .^ 2;
    case 'efftanh'
        df = 1.7159 * 2 / 3 * (1 - (f / 1.7159) .^ 2);
    case 'softplus'
        df = 1 - exp(- f);
    case 'relu'
        df = double(f >= 0);
    case 'prelu'
        alpha = 0.01;
        df = double(f >= 0) + alpha * double(f < 0);
    case 'selu'
        alpha = 1.67326;
        lambda = 1.0507;
        df = lambda * (double(f >= 0) + (alpha + f / lambda) .* (f < 0));
    case 'softmax'
        df = f .* (1 - f); % diagonal derivatives
end

end