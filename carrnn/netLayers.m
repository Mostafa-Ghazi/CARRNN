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

% Custom deep network layers

function lgraph = netLayers(NetParam)

% Output mode for network training
if NetParam.objective == "regression"
    outputMode = 'sequence';
elseif NetParam.objective == "classification"
    outputMode = 'last';
end

% Defining the layers
input = sequenceInputLayer(2 * NetParam.inputSize + 1, 'Name', 'input');
fold = sequenceFoldingLayer('Name', 'fold');
data = splitLayer(1, NetParam.inputSize, 'data');
lags = splitLayer(NetParam.inputSize + 1, 2 * NetParam.inputSize, 'lags');
intervals = splitLayer(2 * NetParam.inputSize + 1, 2 * NetParam.inputSize + 1, 'intervals');
data_lags = concatenationLayer(1, 2, 'Name', 'data_lags');
filled = carLayer(NetParam.inputSize, [], 'filled');
filled = setL2Factor(filled, 'Weights', 0.9);
filled = setLearnRateFactor(filled, 'Weights', 1.1);
filled_intervals = concatenationLayer(1, 2, 'Name', 'filled_intervals');
irregular = carLayer(NetParam.inputSize, NetParam.tau, 'irregular');
irregular = setL2Factor(irregular, 'Weights', 0.9);
irregular = setLearnRateFactor(irregular, 'Weights', 1.1);
unfold = sequenceUnfoldingLayer('Name', 'unfold');
switch NetParam.netType
    case 'gru'
        future = gruLayer(NetParam.hiddenSize, 'OutputMode', outputMode, 'Name', 'future');
    case 'lstm'
        future = lstmLayer(NetParam.hiddenSize, 'OutputMode', outputMode, 'Name', 'future');
    case 'bilstm'
        future = bilstmLayer(NetParam.hiddenSize, 'OutputMode', outputMode, 'Name', 'future');
end
dropped = dropoutLayer(0.01, 'Name', 'dropped');
output = fullyConnectedLayer(NetParam.outputSize, 'Name', 'output');
switch NetParam.objective
    case 'regression'
        errors = outputRegressionLayer('errors');
    case 'classification'
        scores = softmaxLayer('Name', 'scores');
        errors = outputClassificationLayer('errors');
end

% Connecting the layers
lgraph = addLayers(layerGraph(), [input; fold]);
lgraph = addLayers(lgraph, data);
lgraph = connectLayers(lgraph, 'fold/out', 'data');
lgraph = addLayers(lgraph, lags);
lgraph = connectLayers(lgraph, 'fold/out', 'lags');
lgraph = addLayers(lgraph, intervals);
lgraph = connectLayers(lgraph, 'fold/out', 'intervals');
lgraph = addLayers(lgraph, [data_lags; filled]);
lgraph = connectLayers(lgraph, 'data', 'data_lags/in1');
lgraph = connectLayers(lgraph, 'lags', 'data_lags/in2');
if NetParam.objective == "regression"
    lgraph = addLayers(lgraph, [filled_intervals; irregular; unfold; future; dropped; output; errors]);
elseif NetParam.objective == "classification"
    lgraph = addLayers(lgraph, [filled_intervals; irregular; unfold; future; dropped; output; scores; errors]);
end
lgraph = connectLayers(lgraph, 'filled', 'filled_intervals/in1');
lgraph = connectLayers(lgraph, 'intervals', 'filled_intervals/in2');
lgraph = connectLayers(lgraph, 'fold/miniBatchSize', 'unfold/miniBatchSize');

end