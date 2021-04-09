function performance = netPerformance(y, s, metric)

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

% Measuring the network prediction performance

% y: target array [numSamples x 1]
% s: predicted array [numSamples x 1]
% metric: evaluation metric

switch metric
    
    case 'MSE' % mean square error
        sampleIDX = find(~cellfun(@isempty, s)); % indices of the available predictions
        y = cellfun(@(x) x(:, 2 : end), y, 'UniformOutput', false); % Only keeping target points corresponding to predicted points
        predError = cellfun(@(x,z) (x(:) - z(:)), y(sampleIDX), s(sampleIDX), 'UniformOutput', false); % difference between actual and predicted values
        performance = nanmean(cat(1, predError{:}) .^ 2);
    case 'RMSE' % root mean square error
        sampleIDX = find(~cellfun(@isempty, s)); % indices of the available predictions
        y = cellfun(@(x) x(:, 2 : end), y, 'UniformOutput', false); % Only keeping target points corresponding to predicted points
        predError = cellfun(@(x,z) (x(:) - z(:)), y(sampleIDX), s(sampleIDX), 'UniformOutput', false); % difference between actual and predicted values
        performance = sqrt(nanmean(cat(1, predError{:}) .^ 2));
    case 'MAE' % mean absolute error
        sampleIDX = find(~cellfun(@isempty, s)); % indices of the available predictions
        y = cellfun(@(x) x(:, 2 : end), y, 'UniformOutput', false); % Only keeping target points corresponding to predicted points
        predError = cellfun(@(x,z) (x(:) - z(:)), y(sampleIDX), s(sampleIDX), 'UniformOutput', false); % difference between actual and predicted values
        performance = nanmean(abs(cat(1, predError{:})));
    case 'MAPE' % mean absolute percentage error
        sampleIDX = find(~cellfun(@isempty, s)); % indices of the available predictions
        y = cellfun(@(x) reshape(x(:, 2 : end), [], 1), y, 'UniformOutput', false); % Only keeping vectorized target points corresponding to predicted points
        s = cellfun(@(x) x(:), s, 'UniformOutput', false); % vectorized predicted values
        y = y(sampleIDX); s = s(sampleIDX); % removing empty rows
        performance = nanmean(abs(cat(1, y{:}) - cat(1, s{:})) ./ abs(cat(1, y{:})));
    case 'MRAE' % mean relative absolute error
        sampleIDX = find(~cellfun(@isempty, s)); % indices of the available predictions
        y = cellfun(@(x) reshape(x(:, 2 : end), [], 1), y, 'UniformOutput', false); % Only keeping vectorized target points corresponding to predicted points
        s = cellfun(@(x) x(:), s, 'UniformOutput', false); % vectorized predicted values
        y = y(sampleIDX); s = s(sampleIDX); % removing empty rows
        performance = nanmean(abs(cat(1, y{:}) - cat(1, s{:})) ./ abs(cat(1, y{:}) - nanmean(cat(1, y{:}))));
        
end

end
