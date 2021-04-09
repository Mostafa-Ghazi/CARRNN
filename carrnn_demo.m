
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

% Training, validation, and testing the proposed model for biomarker value prediction using a simulated data

restoredefaultpath
close all
clear
clc

% Input data file
input_data = './data/simulation_data.csv';

% Setting network layer parameters
NetParam.inputSize = 4; % number of input layer nodes
NetParam.hiddenSize = 8; % number of hidden layer nodes
NetParam.outputSize = 4; % number of output layer nodes (features for regression and classes for classification)
NetParam.sigm_g = 'sigmoid'; % activation function for the gates (typically sigmoid)
NetParam.sigm_c = 'tanh'; % activation function for the cell input (typically tanh)
NetParam.sigm_h = 'tanh'; % activation function for the hidden layer input (typically tanh)
NetParam.sigm_y = 'identity'; % activation function for the network output (identity for regression and softmax for classification)
NetParam.netType = 'gru'; % base recursive network type

% Setting optimization parameters
NetParam.tau = 0.25; % regular time step
NetParam.regularizationMethod = 'l2norm'; % weights regularization method
NetParam.gradientThresholdMethod = 'l2norm'; % gradients thresholding method
NetParam.gradientThreshold = 0.5; % gradients clipping threshold
NetParam.initialLearnRate = 0.01; % base learning rate
NetParam.learnRateSchedule = 'time-based';
NetParam.learnRateDropPeriod = 50;
NetParam.learnRateDropFactor = 0.9;
NetParam.learnRateDecay = 0.001;
NetParam.optimType = 'momentum';
NetParam.momentum = 0.9; % momentum weight
NetParam.weightDecay = 5e-5; % weights regularization factor
NetParam.gradientDecayFactor = 0.85;
NetParam.squaredGradientDecayFactor = 0.95;
NetParam.miniBatchSize = 20; % number of samples in each mini-batch
evalMetric = 'MAE'; % evaluation metric
maxEpochs = 100; % maximum number of epochs
ratioValid = 0.2; % proportion of validation samples to entire data
patienceIterations = 10; % number of iterations to wait for early stopping

%% Data Preparation

% Reading the longitudinal data
data = readtable(input_data);
addpath('./carrnn'); % add path to the source codes

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

%% Network Training and Validation

% Randomly sampling training and validation subsets
rng(0); % random number generation seed for reproducibility
numSampleValid = ceil(numSampleTrain * ratioValid); % number of validation samples
rndIDX = randperm(numSampleTrain); % shuffled indices (no replacement)
validIDX = rndIDX(1 : numSampleValid); % indices of validation samples
trainIDX = rndIDX(numSampleValid + 1 : end); % indices of training samples

% Initialization of the network
numIterations = maxEpochs * ceil((numSampleTrain - numSampleValid) / NetParam.miniBatchSize); % batches
modelTrain = netInitialize(NetParam, 'parameters'); % parameter initialization
modelTrain.iteration = 0; % iteration number
modelTrain.learnRate = modelTrain.initialLearnRate; % initial learning rate
modelTrain.lossTrain = zeros(numIterations, 1); % training loss
modelTrain.lossValid = zeros(numIterations, 1); % validation loss
patienceLosses = Inf(1, patienceIterations); % previously smallest losses to compare
earlyStop = false; % early stopping flag

% Displaying training progress
figure, subplot(2, 1, 1), xlabel('Iteration'), ylabel(evalMetric), ylim([0, Inf]), grid on;
lineErrorTrain = animatedline('Color', [0, 0.447, 0.741], 'LineStyle', '-', 'LineWidth', 1.5);
lineErrorValid = animatedline('Color', [0.494, 0.184, 0.556], 'LineStyle', '--', 'LineWidth', 1.5);
legend({'Training', 'Validation'}, 'Location', 'northeast'), legend('boxon');
subplot(2, 1, 2), xlabel('Iteration'), ylabel('Loss'), ylim([0, Inf]), grid on;
lineLossTrain = animatedline('Color', [0.85, 0.325, 0.098], 'LineStyle', '-', 'LineWidth', 1.5);
lineLossValid = animatedline('Color', [0.466, 0.674, 0.188], 'LineStyle', '--', 'LineWidth', 1.5);
legend({'Training', 'Validation'}, 'Location', 'northeast'), legend('boxon');

% Loop over iterations
while modelTrain.iteration < numIterations && ~earlyStop
    
    modelTrain.iteration = modelTrain.iteration + 1; % update iteration number
    batchIDX = circshift(trainIDX, NetParam.miniBatchSize); % shifted indices of training samples
    
    % Initializition of the gradients of the loss w.r.t. the learnable parameters
    modelTrain = netInitialize(modelTrain, 'gradients');
    
    % Loop over training mini-batches
    modelTrain.miniBatchSize = NetParam.miniBatchSize;
    modelTrain.miniBatchSizeIrregular = 0; % number of samples with irregularities in each mini-batch
    modelTrain.miniBatchSizeFilled = 0; % number of samples per feature with imputation in each mini-batch
    binnedData = cell(NetParam.miniBatchSize, 1); % aligned training data
    binnedDataPred = cell(NetParam.miniBatchSize, 1); % predicted training data
    for i = 1 : NetParam.miniBatchSize
        [binnedData{i}, binnedStamps, visitInterval] = dataBinning(y1{batchIDX(i)}, stamps1{batchIDX(i)}, modelTrain.tau); % dynamic time binning (aligning the visits)
        modelTrain.sequenceSize = length(visitInterval); % time sequence length (timespan)
        if modelTrain.sequenceSize > 1
            [filledData, featureInterval] = dataFilling(binnedData{i}(1 : K, 1 : end - 1), binnedStamps(:, 1 : end - 1), 'nearest-filling'); % input missing data imputation
            modelTrain = netForward(modelTrain, filledData, featureInterval, visitInterval); % feedforward
            binnedDataPred{i} = modelTrain.outputActive; % predicted values
            [loss, modelTrain] = netLoss(modelTrain, binnedData{i}(1 : K, 2 : end)); % regression loss
            modelTrain.lossTrain(modelTrain.iteration) = modelTrain.lossTrain(modelTrain.iteration) + loss; % normalized loss
            modelTrain = netBackward(modelTrain, featureInterval, visitInterval); % backpropagation
            % modelTrain = netGradients(modelTrain, filledData, binnedData{i}(1 : K, 2 : end), featureInterval, visitInterval, loss); % numerical gradients
            modelTrain.miniBatchSizeIrregular = modelTrain.miniBatchSizeIrregular + (sum(~(visitInterval - modelTrain.tau)) > 0); % any irregular cases
            modelTrain.miniBatchSizeFilled = modelTrain.miniBatchSizeFilled + (sum(featureInterval ~= 0, 2) > 0); % any imputed cases
        else
            modelTrain.miniBatchSize = modelTrain.miniBatchSize - 1;
        end
    end
    modelTrain.lossTrain(modelTrain.iteration) = modelTrain.lossTrain(modelTrain.iteration) / modelTrain.miniBatchSize; % average of the accumulated loss
    modelTrain.errorTrain(modelTrain.iteration) = netPerformance(binnedData, binnedDataPred, evalMetric); % training performance
    
    % Validation performance
    modelTrain.numSampleValid = numSampleValid;
    binnedData = cell(numSampleValid, 1); % aligned validation data
    binnedDataPred = cell(numSampleValid, 1); % predicted validation data
    for i = 1 : numSampleValid
        [binnedData{i}, binnedStamps, visitInterval] = dataBinning(y1{validIDX(i)}, stamps1{validIDX(i)}, modelTrain.tau); % dynamic time binning (aligning the visits)
        modelTrain.sequenceSize = length(visitInterval); % time sequence length (timespan)
        if modelTrain.sequenceSize > 1
            [filledData, featureInterval] = dataFilling(binnedData{i}(1 : K, 1 : end - 1), binnedStamps(:, 1 : end - 1), 'nearest-filling'); % input missing data imputation
            modelTrain = netForward(modelTrain, filledData, featureInterval, visitInterval); % feedforward
            binnedDataPred{i} = modelTrain.outputActive; % predicted values
            loss = netLoss(modelTrain, binnedData{i}(1 : K, 2 : end)); % regression loss
            modelTrain.lossValid(modelTrain.iteration) = modelTrain.lossValid(modelTrain.iteration) + loss; % normalized loss
        else
            modelTrain.numSampleValid = modelTrain.numSampleValid - 1;
        end
    end
    modelTrain.lossValid(modelTrain.iteration) = modelTrain.lossValid(modelTrain.iteration) / modelTrain.numSampleValid; % average of the accumulated loss
    modelTrain.errorValid(modelTrain.iteration) = netPerformance(binnedData, binnedDataPred, evalMetric); % validation performance
    
    if isfinite(patienceIterations)
        patienceLosses = cat(2, patienceLosses, modelTrain.lossValid(modelTrain.iteration)); % append validation losses
        if min(patienceLosses) == patienceLosses(1) % stop updating
            earlyStop = true; % no performance improvements after patience
        else % update network
            patienceLosses(1) = []; % performance is improving
            
            % Updating the learnable parameters and their gradients
            modelTrain = netUpdate(modelTrain, 'gradients'); % update gradients over the mini-batch
            modelTrain = netRegularize(modelTrain); % L1/L2 regularization of the weights
            modelTrain = netThreshold(modelTrain); % clip the gradients of the learnable parameters
            modelTrain = netUpdate(modelTrain, modelTrain.optimType); % update parameters using gradient descent algorithms
            
            % Displaying the training progress
            if modelTrain.iteration == 1
                clearpoints(lineErrorTrain), clearpoints(lineErrorValid), clearpoints(lineLossTrain), clearpoints(lineLossValid); % clear points from animated lines
            end
            addpoints(lineErrorTrain, modelTrain.iteration, modelTrain.errorTrain(modelTrain.iteration)); % proceeding error line
            addpoints(lineErrorValid, modelTrain.iteration, modelTrain.errorValid(modelTrain.iteration)); % proceeding error line
            addpoints(lineLossTrain, modelTrain.iteration, modelTrain.lossTrain(modelTrain.iteration)); % proceeding loss line
            addpoints(lineLossValid, modelTrain.iteration, modelTrain.lossValid(modelTrain.iteration)); % proceeding loss line
            drawnow; % update figure
            
            % Dropping the base learning rate
            if strcmpi(modelTrain.learnRateSchedule, 'piecewise') && mod(modelTrain.iteration, modelTrain.learnRateDropPeriod) == 0
                modelTrain.learnRate = modelTrain.learnRate * modelTrain.learnRateDropFactor;
            elseif strcmpi(modelTrain.learnRateSchedule, 'time-based')
                modelTrain.learnRate = modelTrain.initialLearnRate / (1 + modelTrain.learnRateDecay * modelTrain.iteration);
            end
        end
    end
    
end

%% Network Testing

% Test prediction performance based on all available time points
binnedData = cell(numSampleTest, 1); % aligned test data
binnedDataPred = cell(numSampleTest, 1); % predicted test data
for i = 1 : numSampleTest
    [binnedData{i}, binnedStamps, visitInterval] = dataBinning(y2{i}, stamps2{i}, modelTrain.tau); % dynamic time binning (aligning the visits)
    modelTrain.sequenceSize = length(visitInterval); % time sequence length (timespan)
    if modelTrain.sequenceSize > 0
        [filledData, featureInterval] = dataFilling(binnedData{i}(1 : K, 1 : end - 1), binnedStamps(:, 1 : end - 1), 'nearest-filling'); % input missing data imputation
        modelTrain = netForward(modelTrain, filledData, featureInterval, visitInterval); % feedforward
        binnedDataPred{i} = modelTrain.outputActive; % predicted values
    end
end
errorTest = netPerformance(binnedData, binnedDataPred, evalMetric); % test performance
fprintf('Test MAE = %4.4f \n', errorTest); % display test modeling performance
