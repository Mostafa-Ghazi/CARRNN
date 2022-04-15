
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

% Training and testing the proposed model for biomarker value prediction using a simulated data

restoredefaultpath
close all
clear
clc

% Input data file
input_data = './data/simulation_data.csv';
addpath('./carrnn'); % add path to the source codes

% Setting optimization parameters
NetParam.tau = 0.25; % regular time step
NetParam.optimType = 'adam';
NetParam.momentum = 0.9; % momentum weight for optimType = 'sgdm'
NetParam.regularizationMethod = 'l2norm'; % weights regularization method
NetParam.weightDecay = 1e-3; % weights regularization factor
NetParam.gradientThresholdMethod = 'l2norm'; % gradients thresholding method
NetParam.gradientThreshold = 0.1; % gradients clipping threshold
NetParam.gradientDecayFactor = 0.85; % for optimType = 'adam'
NetParam.squaredGradientDecayFactor = 0.95; % for optimType = 'adam' or 'rmsprop'
NetParam.initialLearnRate = 1e-3; % base learning rate
NetParam.learnRateSchedule = 'piecewise';
NetParam.learnRateDropPeriod = 100;
NetParam.learnRateDropFactor = 0.9;
maxEpochs = 100; % maximum number of epochs
NetParam.miniBatchSize = 20; % number of samples in each mini-batch
evalMetric = 'MAE'; % evaluation metric
verbose = true;
plots = 'training-progress';
NetParam.executionEnvironment = 'gpu';
checkpointPath = './checkpoints';
if ~exist(checkpointPath, 'dir')
    mkdir(checkpointPath)
end
options = trainingOptions(NetParam.optimType, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', NetParam.miniBatchSize, ...
    'L2Regularization', NetParam.weightDecay, ...
    'GradientThresholdMethod', NetParam.gradientThresholdMethod, ...
    'GradientThreshold', NetParam.gradientThreshold, ...
    'GradientDecayFactor', NetParam.gradientDecayFactor, ...
    'SquaredGradientDecayFactor', NetParam.squaredGradientDecayFactor, ...
    'InitialLearnRate', NetParam.initialLearnRate, ...
    'LearnRateSchedule', NetParam.learnRateSchedule, ...
    'LearnRateDropPeriod', NetParam.learnRateDropPeriod, ...
    'LearnRateDropFactor', NetParam.learnRateDropFactor, ...
    'Shuffle', 'never', ...
    'SequenceLength', 'shortest', ...
    'SequencePaddingDirection', 'right', ...
    'SequencePaddingValue', realmax, ...
    'ExecutionEnvironment', NetParam.executionEnvironment, ...
    'Verbose', verbose, ...
    'Plots', plots, ...
    'CheckpointPath', checkpointPath);

% Setting network layer parameters
NetParam.inputSize = 4; % number of input layer nodes
NetParam.hiddenSize = 8; % number of hidden layer nodes
NetParam.outputSize = 4; % number of output layer nodes (features for regression and classes for classification)
NetParam.netType = 'gru'; % base recursive network type
NetParam.objective = 'regression'; % network objective (regression or classification)
lgraph = netLayers(NetParam); % network graph
analyzeNetwork(lgraph); % visualized network

%% Data Preparation

% Reading the longitudinal data
data = readtable(input_data);

% Extracting desired fields from the longitudinal data
samples_vec = data{:, 'SubjectID'}; % subject IDs (samples)
labels_vec = data{:, 'Label'}; % visiting status of subjects
dates_vec = data{:, 'Age'}; % visiting ages (dates) of subjects
features = setdiff(data.Properties.VariableNames, {'SubjectID', 'Label', 'Age'}); % feature (biomarker) names
y_vec = data{:, ismember(data.Properties.VariableNames, features)}; % feature values (measurements)
features = cat(2, features, 'Labels'); % feature names appended with class labels

% Removing data lacking visiting date information and vice versa
y_vec(isnan(dates_vec), :) = NaN;
dates_vec(~sum(~isnan(y_vec), 2)) = [];
labels_vec(~sum(~isnan(y_vec), 2)) = [];
y_vec(~sum(~isnan(y_vec), 2), :) = [];

% Converting categorical labels to numeric ones
labels_vec_num = NaN(size(labels_vec)); % numerical labels
labels_vec_num(~cellfun(@isempty, labels_vec)) = findgroups(labels_vec(~cellfun(@isempty, labels_vec))) - 2; % {'AD', 'CN', 'MCI'} -> [-1, 0, 1]

% Arrenging data based on temporal ordering of visits
samples = unique(samples_vec, 'sorted'); % unique subjects IDs
I = length(samples); % number of samples (subjects)
K = length(features); % number of features (biomarkers+labels)
J = zeros(I, 1); % number of timestamps (visits)
y = cell(I, 1); % subject features
stamps = cell(I, 1); % visiting timestamps
for i = 1 : I
    subjectIDX = find(samples_vec == samples(i)); % indices of the subject's measurements
    [stamps{i}, sortIDX] = sort(dates_vec(subjectIDX), 'ascend'); % sorted values
    J(i) = numel(stamps{i}); % number of subject's visits
    if J(i) > 1 % subject with at least two distinct time points
        y{i} = y_vec(subjectIDX(sortIDX), :)'; % per visit features of the subject
        y{i} = cat(1, y{i}, labels_vec_num(subjectIDX(sortIDX))'); % features appended with labels
    end
    stamps{i} = stamps{i}' - min(stamps{i}); % distinct timestamps starting from zero
end
idx_empty = (cellfun(@isempty, y) | cellfun(@isempty, stamps)); % logical indices of empty rows
I = I - sum(idx_empty); % updated number of subjects
J(idx_empty) = []; % updated number of visits
y(idx_empty) = []; % updated measurements
stamps(idx_empty) = []; % updated visiting timestamps
samples(idx_empty) = []; % updated subject IDs

% Splitting data to training (train+valid) and test subsets
rng(0); % random number generation seed
ratioTest = 0.2; % proportion of test samples to entire data
testIDX = randperm(I, ceil(ratioTest * I)); % random indices of test samples
trainIDX = find(~ismember(1 : I, testIDX)); % random indices of training samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numSampleTrain = numel(trainIDX); % number of training samples
J1 = J(trainIDX); % number of training visits
y1 = y(trainIDX); % training measurements
stamps1 = stamps(trainIDX); % training timestamps
samples1 = samples(trainIDX); % training subject IDs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numSampleTest = numel(testIDX); % number of test samples
J2 = J(testIDX); % number of test visits
y2 = y(testIDX); % test measurements
stamps2 = stamps(testIDX); % test timestamps
samples2 = samples(testIDX); % test subject IDs

% Standardizing measurements (except for the appended labels)
y_mean = nanmean(cat(2, y1{:}), 2); % average measurements
y_std = nanstd(cat(2, y1{:}), 1, 2); % standard deviation of measurements
for i = 1 : numSampleTrain
    for k = 1 : K - 1
        y1{i}(k, :) = (y1{i}(k, :) - y_mean(k)) / y_std(k); % zero-mean and unit variance
    end
end
for i = 1 : numSampleTest
    for k = 1 : K - 1
        y2{i}(k, :) = (y2{i}(k, :) - y_mean(k)) / y_std(k); % zero-mean and unit variance
    end
end

% Normalizing time intervals
stamps_min = min(cat(2, stamps1{:})); % minimum timestamp
stamps_max = max(cat(2, stamps1{:})); % maximum timestamp
stamps_mean = mean(cat(2, stamps1{:})); % average timestamp
stamps_iqr = quantile(cat(2, stamps1{:}), [0.25 0.75]); % minimum and maximum values of the interquartile range of timestamps
for i = 1 : numSampleTrain
    stamps1{i} = (stamps1{i} - stamps_min) / (stamps_iqr(2) - stamps_iqr(1)); % almost within [0 1]
end
for i = 1 : numSampleTest
    stamps2{i} = (stamps2{i} - stamps_min) / (stamps_iqr(2) - stamps_iqr(1)); % almost within [0 1]
end

% Acquiring time intervals and their statistics
dt_train = cellfun(@(x) diff(x, 1, 2), stamps1, 'UniformOutput', false); % time intervals between consecutive visits of training subjects
dt_test = cellfun(@(x) diff(x, 1, 2), stamps2, 'UniformOutput', false); % time intervals between consecutive visits of test subjects
dt_min = min(cat(2, dt_train{:})); % minimum time interval
dt_max = max(cat(2, dt_train{:})); % maximum time interval
dt_mean = nanmean(cat(2, dt_train{:}), 2); % average time interval
dt_std = nanstd(cat(2, dt_train{:}), 1, 2); % standard deviation of time intervals
dt_iqr = quantile(cat(2, dt_train{:}), [0.25 0.75]); % minimum and maximum values of the interquartile range of the time intervals

%% Dataset Arrangement

% Training data binning
dataTrain = cell(numSampleTrain, 1); % binned training data
stampsTrain = cell(numSampleTrain, 1); % binned training timestamps
XTrain = cell(numSampleTrain, 1); % aligned training input
YTrain = cell(numSampleTrain, 1); % training target
for i = 1 : numSampleTrain
    [dataTrain{i}, stampsTrain{i}, visitInterval] = dataBinning(y1{i}, stamps1{i}, NetParam.tau); % time binning (aligning the visits)
    if length(visitInterval) > 1
        [filledData, featureInterval] = dataFilling(dataTrain{i}(:, 1 : end - 1), stampsTrain{i}(:, 1 : end - 1), 'nearest-filling'); % missing data imputation
        XTrain{i} = [filledData; featureInterval; visitInterval];
        YTrain{i} = dataTrain{i}(:, 2 : end);
    end
end
numSampleTrain = numSampleTrain - sum(cellfun(@isempty, XTrain)); % updated number of training samples
YTrain(cellfun(@isempty, XTrain)) = []; % updated training target
dataTrain(cellfun(@isempty, XTrain)) = []; % updated training data
stampsTrain(cellfun(@isempty, XTrain)) = []; % updated training timestamps
XTrain(cellfun(@isempty, XTrain)) = []; % updated training input

% Test data binning
dataTest = cell(numSampleTest, 1); % binned test data
stampsTest = cell(numSampleTest, 1); % binned test timestamps
XTest = cell(numSampleTest, 1); % aligned test input
YTest = cell(numSampleTest, 1); % test target
for i = 1 : numSampleTest
    [dataTest{i}, stampsTest{i}, visitInterval] = dataBinning(y2{i}, stamps2{i}, NetParam.tau); % time binning (aligning the visits)
    if length(visitInterval) > 1
        [filledData, featureInterval] = dataFilling(dataTest{i}(:, 1 : end - 1), stampsTest{i}(:, 1 : end - 1), 'nearest-filling'); % missing data imputation
        XTest{i} = [filledData; featureInterval; visitInterval];
        YTest{i} = dataTest{i}(:, 2 : end);
    end
end
numSampleTest = numSampleTest - sum(cellfun(@isempty, XTest)); % updated number of test samples
YTest(cellfun(@isempty, XTest)) = []; % updated test target
dataTest(cellfun(@isempty, XTest)) = []; % updated test data
stampsTest(cellfun(@isempty, XTest)) = []; % updated test timestamps
XTest(cellfun(@isempty, XTest)) = []; % updated test input

% Replacing available NaN values
XTrain = cellfun(@(x) fillmissing(x, 'constant', realmax), XTrain, 'UniformOutput', false);
YTrain = cellfun(@(x) fillmissing(x, 'constant', realmax), YTrain, 'UniformOutput', false);
XTest = cellfun(@(x) fillmissing(x, 'constant', realmax), XTest, 'UniformOutput', false);
YTest = cellfun(@(x) fillmissing(x, 'constant', realmax), YTest, 'UniformOutput', false);

% Sorting sequences by length
rndIDXTrain = randperm(numel(XTrain));
sequenceLengths = cellfun(@(X) size(X, 2), XTrain(rndIDXTrain));
[~, sortIDXTrain] = sort(sequenceLengths);
XTrain = XTrain(rndIDXTrain(sortIDXTrain));
YTrain = YTrain(rndIDXTrain(sortIDXTrain));
dataTrain = dataTrain(rndIDXTrain(sortIDXTrain));
stampsTrain = stampsTrain(rndIDXTrain(sortIDXTrain));
rndIDXTest = randperm(numel(XTest));
sequenceLengths = cellfun(@(X) size(X, 2), XTest(rndIDXTest));
[~, sortIDXTest] = sort(sequenceLengths);
XTest = XTest(rndIDXTest(sortIDXTest));
YTest = YTest(rndIDXTest(sortIDXTest));
dataTest = dataTest(rndIDXTest(sortIDXTest));
stampsTest = stampsTest(rndIDXTest(sortIDXTest));

%% Network Training

% Training the network
[NetParam.dlnet, NetParam.performance] = trainNetwork(XTrain, YTrain, lgraph, options);

%% Network Testing

% Test prediction performance based on all available time points
% Test prediction performance based on all available time points
[~, YTestPred] = predictAndUpdateState(NetParam.dlnet, XTest, ...
    'MiniBatchSize', numSampleTest, ...
    'SequenceLength', 'longest', ...
    'SequencePaddingDirection', 'right', ...
    'SequencePaddingValue', realmax, ...
    'Acceleration', 'auto', ...
    'ExecutionEnvironment', NetParam.executionEnvironment);
errorTest = netPerformance(cellfun(@(x) [NaN(size(x, 1), 1), (x + 0 ./ (x ~= realmax))], YTest, 'UniformOutput', false), YTestPred, evalMetric); % test performance
fprintf(['Test ' evalMetric ' = %4.4f \n'], errorTest); % display test modeling performance
