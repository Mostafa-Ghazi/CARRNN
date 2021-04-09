function NetParam = netBackward(NetParam, dt_x, dt_xy)

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

% Backpropagation of the errors through the network and calculating the gradients

% NetParam: structure including the network and optimization parameters, memory blocks, and gradients
% dt_x: time intervals of the input features [inputSize x sequenceSize]
% dt_xy: time intervals of the aligned visits [1 x sequenceSize]

switch NetParam.netType
    
    case 'rnn'
        
        % Initialization of the gradients of the memory blocks before and after activation
        NetParam.deltaOutput = zeros(NetParam.outputSize, NetParam.sequenceSize); % derivative of loss w.r.t. output [dL / dy]
        NetParam.deltaHiddenIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. irregular hidden
        NetParam.deltaHiddenActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired hidden
        NetParam.deltaHidden = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. hidden [dL / dh]
        NetParam.deltaInputFilledScaled = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. scaled imputed input
        NetParam.deltaInputFilled = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. imputed input
        NetParam.deltaInput = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. input [dL / dx]
        
        % Gradients of the memory blocks before and after activation in the last timestamp
        NetParam.deltaOutput(:, end) = NetParam.deltaOutputActive(:, end) .* activationDerivativeOut(NetParam.outputActive(:, end), NetParam.sigm_y);
        % NetParam.deltaOutput(:, end) = NetParam.deltaOutputActive(:, end) .* activationDerivativeIn(NetParam.output(:, end), NetParam.sigm_y);
        NetParam.deltaHiddenIrregular(:, end) = NetParam.Wy' * NetParam.deltaOutput(:, end);
        NetParam.deltaHiddenActive(:, end) = (eye(NetParam.hiddenSize) + (dt_xy(end) - NetParam.tau) * NetParam.Ph') * NetParam.deltaHiddenIrregular(:, end);
        NetParam.deltaHidden(:, end) = NetParam.deltaHiddenActive(:, end) .* activationDerivativeOut(NetParam.hiddenActive(:, end), NetParam.sigm_h);
        % NetParam.deltaHidden(:, end) = NetParam.deltaHiddenActive(:, end) .* activationDerivativeIn(NetParam.hidden(:, end), NetParam.sigm_h);
        NetParam.deltaInputFilledScaled(:, end) = NetParam.Wh' * NetParam.deltaHidden(:, end);
        NetParam.deltaInputFilled(:, end) = NetParam.deltaInputFilledScaled(:, end) * NetParam.inputFactor(end);
        NetParam.deltaInput(:, end) = (1 + dt_x(:, end) .* NetParam.Px) .* NetParam.deltaInputFilled(:, end);
        
        % Backpropagation through time
        if NetParam.sequenceSize > 1
            for k = NetParam.sequenceSize - 1 : - 1 : 1
                NetParam.deltaOutput(:, k) = NetParam.deltaOutputActive(:, k) .* activationDerivativeOut(NetParam.outputActive(:, k), NetParam.sigm_y);
                % NetParam.deltaOutput(:, k) = NetParam.deltaOutputActive(:, k) .* activationDerivativeIn(NetParam.output(:, k), NetParam.sigm_y);
                NetParam.deltaHiddenIrregular(:, k) = NetParam.Wy' * NetParam.deltaOutput(:, k) + NetParam.Uh' * NetParam.deltaHidden(:, k + 1);
                NetParam.deltaHiddenActive(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Ph') * NetParam.deltaHiddenIrregular(:, k);
                NetParam.deltaHidden(:, k) = NetParam.deltaHiddenActive(:, k) .* activationDerivativeOut(NetParam.hiddenActive(:, k), NetParam.sigm_h);
                % NetParam.deltaHidden(:, k) = NetParam.deltaHiddenActive(:, k) .* activationDerivativeIn(NetParam.hidden(:, k), NetParam.sigm_h);
                NetParam.deltaInputFilledScaled(:, k) = NetParam.Wh' * NetParam.deltaHidden(:, k);
                NetParam.deltaInputFilled(:, k) = NetParam.deltaInputFilledScaled(:, k) * NetParam.inputFactor(k);
                NetParam.deltaInput(:, k) = (1 + dt_x(:, k) .* NetParam.Px) .* NetParam.deltaInputFilled(:, k);
            end
        end
        
        % Gradients of the learnable parameters across all timestamps
        NetParam.dL_Wy = NetParam.dL_Wy + NetParam.deltaOutput * NetParam.hiddenIrregular';
        NetParam.dL_by = NetParam.dL_by + sum(NetParam.deltaOutput, 2);
        NetParam.dL_Wh = NetParam.dL_Wh + NetParam.deltaHidden * NetParam.inputFilledScaled';
        NetParam.dL_Uh = NetParam.dL_Uh + NetParam.deltaHidden(:, 2 : end) * NetParam.hiddenIrregular(:, 1 : end - 1)' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_bh = NetParam.dL_bh + sum(NetParam.deltaHidden, 2);
        NetParam.dL_Ph = NetParam.dL_Ph + (NetParam.deltaHiddenIrregular * diag(dt_xy - NetParam.tau)) * NetParam.hiddenActive' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Ch = NetParam.dL_Ch + NetParam.deltaHiddenIrregular * (dt_xy - NetParam.tau)' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Px = NetParam.dL_Px + sum(NetParam.deltaInputFilled .* dt_x .* NetParam.input, 2) * NetParam.sequenceSize ./ (sum(dt_x ~= 0, 2) + 1e-8);
        NetParam.dL_Cx = NetParam.dL_Cx + sum(NetParam.deltaInputFilled .* dt_x, 2) * NetParam.sequenceSize ./ (sum(dt_x ~= 0, 2) + 1e-8);
        
    case 'lstm'
        
        % Initialization of the gradients of the memory blocks before and after activation
        NetParam.deltaOutput = zeros(NetParam.outputSize, NetParam.sequenceSize); % derivative of loss w.r.t. output [dL / dy]
        NetParam.deltaHiddenIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. irregular hidden
        NetParam.deltaHidden = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. hidden [dL / dh]
        NetParam.deltaOutputGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired output gate
        NetParam.deltaOutputGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. output gate [dL / do]
        NetParam.deltaCellStateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired cell state
        NetParam.deltaCellStateIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. irregular cell state
        NetParam.deltaCellState = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. cell state [dL / dc]
        NetParam.deltaModulationGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired modulation gate
        NetParam.deltaModulationGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. modulation gate [dL / dz]
        NetParam.deltaInputGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired input gate
        NetParam.deltaInputGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. input gate [dL / di]
        NetParam.deltaForgetGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired forget gate
        NetParam.deltaForgetGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. forget gate [dL / df]
        NetParam.deltaInputFilledScaled = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. scaled imputed input
        NetParam.deltaInputFilled = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. imputed input
        NetParam.deltaInput = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. input [dL / dx]
        
        % Gradients of the memory blocks before and after activation in the last timestamp
        NetParam.deltaOutput(:, end) = NetParam.deltaOutputActive(:, end) .* activationDerivativeOut(NetParam.outputActive(:, end), NetParam.sigm_y);
        % NetParam.deltaOutput(:, end) = NetParam.deltaOutputActive(:, end) .* activationDerivativeIn(NetParam.output(:, end), NetParam.sigm_y);
        NetParam.deltaHiddenIrregular(:, end) = NetParam.Wy' * NetParam.deltaOutput(:, end);
        NetParam.deltaHidden(:, end) = (eye(NetParam.hiddenSize) + (dt_xy(end) - NetParam.tau) * NetParam.Ph') * NetParam.deltaHiddenIrregular(:, end);
        NetParam.deltaOutputGateActive(:, end) = NetParam.deltaHidden(:, end) .* NetParam.cellStateActive(:, end);
        NetParam.deltaOutputGate(:, end) = NetParam.deltaOutputGateActive(:, end) .* activationDerivativeOut(NetParam.outputGateActive(:, end), NetParam.sigm_g);
        % NetParam.deltaOutputGate(:, end) = NetParam.deltaOutputGateActive(:, end) .* activationDerivativeIn(NetParam.outputGate(:, end), NetParam.sigm_g);
        NetParam.deltaCellStateActive(:, end) = NetParam.deltaHidden(:, end) .* NetParam.outputGateActive(:, end);
        NetParam.deltaCellStateIrregular(:, end) = NetParam.Vo' * NetParam.deltaOutputGate(:, end);
        NetParam.deltaCellState(:, end) = (eye(NetParam.hiddenSize) + (dt_xy(end) - NetParam.tau) * NetParam.Pc') * NetParam.deltaCellStateIrregular(:, end) + NetParam.deltaCellStateActive(:, end) .* activationDerivativeOut(NetParam.cellStateActive(:, end), NetParam.sigm_h);
        % NetParam.deltaCellState(:, end) = (eye(NetParam.hiddenSize) + (dt_xy(end) - NetParam.tau) * NetParam.Pc') * NetParam.deltaCellStateIrregular(:, end) + NetParam.deltaCellStateActive(:, end) .* activationDerivativeIn(NetParam.cellState(:, end), NetParam.sigm_h);
        NetParam.deltaModulationGateActive(:, end) = NetParam.deltaCellState(:, end) .* NetParam.inputGateActive(:, end);
        NetParam.deltaModulationGate(:, end) = NetParam.deltaModulationGateActive(:, end) .* activationDerivativeOut(NetParam.modulationGateActive(:, end), NetParam.sigm_c);
        % NetParam.deltaModulationGate(:, end) = NetParam.deltaModulationGateActive(:, end) .* activationDerivativeIn(NetParam.modulationGate(:, end), NetParam.sigm_c);
        NetParam.deltaInputGateActive(:, end) = NetParam.deltaCellState(:, end) .* NetParam.modulationGateActive(:, end);
        NetParam.deltaInputGate(:, end) = NetParam.deltaInputGateActive(:, end) .* activationDerivativeOut(NetParam.inputGateActive(:, end), NetParam.sigm_g);
        % NetParam.deltaInputGate(:, end) = NetParam.deltaInputGateActive(:, end) .* activationDerivativeIn(NetParam.inputGate(:, end), NetParam.sigm_g);
        if NetParam.sequenceSize > 1
            NetParam.deltaForgetGateActive(:, end) = NetParam.deltaCellState(:, end) .* NetParam.cellStateIrregular(:, end - 1);
            NetParam.deltaForgetGate(:, end) = NetParam.deltaForgetGateActive(:, end) .* activationDerivativeOut(NetParam.forgetGateActive(:, end), NetParam.sigm_g);
            % NetParam.deltaForgetGate(:, end) = NetParam.deltaForgetGateActive(:, end) .* activationDerivativeIn(NetParam.forgetGate(:, end), NetParam.sigm_g);
        end
        NetParam.deltaInputFilledScaled(:, end) = NetParam.Wf' * NetParam.deltaForgetGate(:, end) + NetParam.Wi' * NetParam.deltaInputGate(:, end) + NetParam.Wz' * NetParam.deltaModulationGate(:, end) + NetParam.Wo' * NetParam.deltaOutputGate(:, end);
        NetParam.deltaInputFilled(:, end) = NetParam.deltaInputFilledScaled(:, end) * NetParam.inputFactor(end);
        NetParam.deltaInput(:, end) = (1 + dt_x(:, end) .* NetParam.Px) .* NetParam.deltaInputFilled(:, end);
        
        % Backpropagation through time
        if NetParam.sequenceSize > 1
            for k = NetParam.sequenceSize - 1 : - 1 : 1
                NetParam.deltaOutput(:, k) = NetParam.deltaOutputActive(:, k) .* activationDerivativeOut(NetParam.outputActive(:, k), NetParam.sigm_y);
                % NetParam.deltaOutput(:, k) = NetParam.deltaOutputActive(:, k) .* activationDerivativeIn(NetParam.output(:, k), NetParam.sigm_y);
                NetParam.deltaHiddenIrregular(:, k) = NetParam.Wy' * NetParam.deltaOutput(:, k) + NetParam.Uf' * NetParam.deltaForgetGate(:, k + 1) + NetParam.Ui' * NetParam.deltaInputGate(:, k + 1) + NetParam.Uz' * NetParam.deltaModulationGate(:, k + 1) + NetParam.Uo' * NetParam.deltaOutputGate(:, k + 1);
                NetParam.deltaHidden(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Ph') * NetParam.deltaHiddenIrregular(:, k);
                NetParam.deltaOutputGateActive(:, k) = NetParam.deltaHidden(:, k) .* NetParam.cellStateActive(:, k);
                NetParam.deltaOutputGate(:, k) = NetParam.deltaOutputGateActive(:, k) .* activationDerivativeOut(NetParam.outputGateActive(:, k), NetParam.sigm_g);
                % NetParam.deltaOutputGate(:, k) = NetParam.deltaOutputGateActive(:, k) .* activationDerivativeIn(NetParam.outputGate(:, k), NetParam.sigm_g);
                NetParam.deltaCellStateActive(:, k) = NetParam.deltaHidden(:, k) .* NetParam.outputGateActive(:, k);
                NetParam.deltaCellStateIrregular(:, k) = NetParam.Vf' * NetParam.deltaForgetGate(:, k + 1) + NetParam.Vi' * NetParam.deltaInputGate(:, k + 1) + NetParam.Vo' * NetParam.deltaOutputGate(:, k) + NetParam.deltaCellState(:, k + 1) .* NetParam.forgetGateActive(:, k + 1);
                NetParam.deltaCellState(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Pc') * NetParam.deltaCellStateIrregular(:, k) + NetParam.deltaCellStateActive(:, k) .* activationDerivativeOut(NetParam.cellStateActive(:, k), NetParam.sigm_h);
                % NetParam.deltaCellState(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Pc') * NetParam.deltaCellStateIrregular(:, k) + NetParam.deltaCellStateActive(:, k) .* activationDerivativeIn(NetParam.cellState(:, k), NetParam.sigm_h);
                NetParam.deltaModulationGateActive(:, k) = NetParam.deltaCellState(:, k) .* NetParam.inputGateActive(:, k);
                NetParam.deltaModulationGate(:, k) = NetParam.deltaModulationGateActive(:, k) .* activationDerivativeOut(NetParam.modulationGateActive(:, k), NetParam.sigm_c);
                % NetParam.deltaModulationGate(:, k) = NetParam.deltaModulationGateActive(:, k) .* activationDerivativeIn(NetParam.modulationGate(:, k), NetParam.sigm_c);
                NetParam.deltaInputGateActive(:, k) = NetParam.deltaCellState(:, k) .* NetParam.modulationGateActive(:, k);
                NetParam.deltaInputGate(:, k) = NetParam.deltaInputGateActive(:, k) .* activationDerivativeOut(NetParam.inputGateActive(:, k), NetParam.sigm_g);
                % NetParam.deltaInputGate(:, k) = NetParam.deltaInputGateActive(:, k) .* activationDerivativeIn(NetParam.inputGate(:, k), NetParam.sigm_g);
                if k > 1
                    NetParam.deltaForgetGateActive(:, k) = NetParam.deltaCellState(:, k) .* NetParam.cellStateIrregular(:, k - 1);
                    NetParam.deltaForgetGate(:, k) = NetParam.deltaForgetGateActive(:, k) .* activationDerivativeOut(NetParam.forgetGateActive(:, k), NetParam.sigm_g);
                    % NetParam.deltaForgetGate(:, k) = NetParam.deltaForgetGateActive(:, k) .* activationDerivativeIn(NetParam.forgetGate(:, k), NetParam.sigm_g);
                end
                NetParam.deltaInputFilledScaled(:, k) = NetParam.Wf' * NetParam.deltaForgetGate(:, k) + NetParam.Wi' * NetParam.deltaInputGate(:, k) + NetParam.Wz' * NetParam.deltaModulationGate(:, k) + NetParam.Wo' * NetParam.deltaOutputGate(:, k);
                NetParam.deltaInputFilled(:, k) = NetParam.deltaInputFilledScaled(:, k) * NetParam.inputFactor(k);
                NetParam.deltaInput(:, k) = (1 + dt_x(:, k) .* NetParam.Px) .* NetParam.deltaInputFilled(:, k);
            end
        end
        
        % Gradients of the learnable parameters across all timestamps
        NetParam.dL_Wy = NetParam.dL_Wy + NetParam.deltaOutput * NetParam.hiddenIrregular';
        NetParam.dL_by = NetParam.dL_by + sum(NetParam.deltaOutput, 2);
        NetParam.dL_Wf = NetParam.dL_Wf + NetParam.deltaForgetGate * NetParam.inputFilledScaled';
        NetParam.dL_Uf = NetParam.dL_Uf + NetParam.deltaForgetGate(:, 2 : end) * NetParam.hiddenIrregular(:, 1 : end - 1)' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_Vf = NetParam.dL_Vf + diag(sum(NetParam.deltaForgetGate(:, 2 : end) .* NetParam.cellStateIrregular(:, 1 : end - 1), 2)) * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_bf = NetParam.dL_bf + sum(NetParam.deltaForgetGate, 2);
        NetParam.dL_Wi = NetParam.dL_Wi + NetParam.deltaInputGate * NetParam.inputFilledScaled';
        NetParam.dL_Ui = NetParam.dL_Ui + NetParam.deltaInputGate(:, 2 : end) * NetParam.hiddenIrregular(:, 1 : end - 1)' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_Vi = NetParam.dL_Vi + diag(sum(NetParam.deltaInputGate(:, 2 : end) .* NetParam.cellStateIrregular(:, 1 : end - 1), 2)) * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_bi = NetParam.dL_bi + sum(NetParam.deltaInputGate, 2);
        NetParam.dL_Wz = NetParam.dL_Wz + NetParam.deltaModulationGate * NetParam.inputFilledScaled';
        NetParam.dL_Uz = NetParam.dL_Uz + NetParam.deltaModulationGate(:, 2 : end) * NetParam.hiddenIrregular(:, 1 : end - 1)' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_bz = NetParam.dL_bz + sum(NetParam.deltaModulationGate, 2);
        NetParam.dL_Wo = NetParam.dL_Wo + NetParam.deltaOutputGate * NetParam.inputFilledScaled';
        NetParam.dL_Uo = NetParam.dL_Uo + NetParam.deltaOutputGate(:, 2 : end) * NetParam.hiddenIrregular(:, 1 : end - 1)' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_Vo = NetParam.dL_Vo + diag(sum(NetParam.deltaOutputGate .* NetParam.cellStateIrregular, 2));
        NetParam.dL_bo = NetParam.dL_bo + sum(NetParam.deltaOutputGate, 2);
        NetParam.dL_Ph = NetParam.dL_Ph + (NetParam.deltaHiddenIrregular * diag(dt_xy - NetParam.tau)) * NetParam.hidden' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Ch = NetParam.dL_Ch + NetParam.deltaHiddenIrregular * (dt_xy - NetParam.tau)' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Pc = NetParam.dL_Pc + (NetParam.deltaCellStateIrregular * diag(dt_xy - NetParam.tau)) * NetParam.cellState' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Cc = NetParam.dL_Cc + NetParam.deltaCellStateIrregular * (dt_xy - NetParam.tau)' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Px = NetParam.dL_Px + sum(NetParam.deltaInputFilled .* dt_x .* NetParam.input, 2) * NetParam.sequenceSize ./ (sum(dt_x ~= 0, 2) + 1e-8);
        NetParam.dL_Cx = NetParam.dL_Cx + sum(NetParam.deltaInputFilled .* dt_x, 2) * NetParam.sequenceSize ./ (sum(dt_x ~= 0, 2) + 1e-8);
        
    case 'gru'
        
        % Initialization of the gradients of the memory blocks before and after activation
        NetParam.deltaOutput = zeros(NetParam.outputSize, NetParam.sequenceSize); % derivative of loss w.r.t. output [dL / dy]
        NetParam.deltaHiddenIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. irregular hidden
        NetParam.deltaHidden = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. hidden [dL / dh]
        NetParam.deltaCellStateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired cell state
        NetParam.deltaCellState = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. cell state [dL / dc]
        NetParam.deltaResetGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired reset gate
        NetParam.deltaResetGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. reset gate [dL / di]
        NetParam.deltaUpdateGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. fired update gate
        NetParam.deltaUpdateGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % derivative of loss w.r.t. update gate [dL / df]
        NetParam.deltaInputFilledScaled = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. scaled imputed input
        NetParam.deltaInputFilled = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. imputed input
        NetParam.deltaInput = zeros(NetParam.inputSize, NetParam.sequenceSize); % derivative of loss w.r.t. input [dL / dx]
        
        % Gradients of the memory blocks before and after activation in the last timestamp
        NetParam.deltaOutput(:, end) = NetParam.deltaOutputActive(:, end) .* activationDerivativeOut(NetParam.outputActive(:, end), NetParam.sigm_y);
        % NetParam.deltaOutput(:, end) = NetParam.deltaOutputActive(:, end) .* activationDerivativeIn(NetParam.output(:, end), NetParam.sigm_y);
        NetParam.deltaHiddenIrregular(:, end) = NetParam.Wy' * NetParam.deltaOutput(:, end);
        NetParam.deltaHidden(:, end) = (eye(NetParam.hiddenSize) + (dt_xy(end) - NetParam.tau) * NetParam.Ph') * NetParam.deltaHiddenIrregular(:, end);
        NetParam.deltaCellStateActive(:, end) = NetParam.deltaHidden(:, end) .* (1 - NetParam.updateGateActive(:, end));
        NetParam.deltaCellState(:, end) =  NetParam.deltaCellStateActive(:, end) .* activationDerivativeOut(NetParam.cellStateActive(:, end), NetParam.sigm_h);
        % NetParam.deltaCellState(:, end) = NetParam.deltaCellStateActive(:, end) .* activationDerivativeIn(NetParam.cellState(:, end), NetParam.sigm_h);
        if NetParam.sequenceSize > 1
            NetParam.deltaResetGateActive(:, end) = NetParam.hiddenIrregular(:, end - 1) .* (NetParam.Uc' * NetParam.deltaCellState(:, end));
            NetParam.deltaResetGate(:, end) = NetParam.deltaResetGateActive(:, end) .* activationDerivativeOut(NetParam.resetGateActive(:, end), NetParam.sigm_g);
            % NetParam.deltaResetGate(:, end) = NetParam.deltaResetGateActive(:, end) .* activationDerivativeIn(NetParam.resetGate(:, end), NetParam.sigm_g);
            NetParam.deltaUpdateGateActive(:, end) = NetParam.deltaHidden(:, end) .* (NetParam.hiddenIrregular(:, end - 1) - NetParam.cellStateActive(:, end));
        else
            NetParam.deltaUpdateGateActive(:, end) = - NetParam.deltaHidden(:, end) .* NetParam.cellStateActive(:, end);
        end
        NetParam.deltaUpdateGate(:, end) = NetParam.deltaUpdateGateActive(:, end) .* activationDerivativeOut(NetParam.updateGateActive(:, end), NetParam.sigm_g);
        % NetParam.deltaUpdateGate(:, end) = NetParam.deltaUpdateGateActive(:, end) .* activationDerivativeIn(NetParam.updateGate(:, end), NetParam.sigm_g);
        NetParam.deltaInputFilledScaled(:, end) = NetParam.Wz' * NetParam.deltaUpdateGate(:, end) + NetParam.Wr' * NetParam.deltaResetGate(:, end) + NetParam.Wc' * NetParam.deltaCellState(:, end);
        NetParam.deltaInputFilled(:, end) = NetParam.deltaInputFilledScaled(:, end) * NetParam.inputFactor(end);
        NetParam.deltaInput(:, end) = (1 + dt_x(:, end) .* NetParam.Px) .* NetParam.deltaInputFilled(:, end);
        
        % Backpropagation through time
        if NetParam.sequenceSize > 1
            for k = NetParam.sequenceSize - 1 : - 1 : 1
                NetParam.deltaOutput(:, k) = NetParam.deltaOutputActive(:, k) .* activationDerivativeOut(NetParam.outputActive(:, k), NetParam.sigm_y);
                % NetParam.deltaOutput(:, k) = NetParam.deltaOutputActive(:, k) .* activationDerivativeIn(NetParam.output(:, k), NetParam.sigm_y);
                NetParam.deltaHiddenIrregular(:, k) = NetParam.Wy' * NetParam.deltaOutput(:, k) + NetParam.Uz' * NetParam.deltaUpdateGate(:, k + 1) + NetParam.Ur' * NetParam.deltaResetGate(:, k + 1) + NetParam.deltaHidden(:, k + 1) .* NetParam.updateGateActive(:, k + 1) + NetParam.resetGateActive(:, k + 1) .* (NetParam.Uc' * NetParam.deltaCellState(:, k + 1));
                NetParam.deltaHidden(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Ph') * NetParam.deltaHiddenIrregular(:, k);
                NetParam.deltaCellStateActive(:, k) = NetParam.deltaHidden(:, k) .* (1 - NetParam.updateGateActive(:, k));
                NetParam.deltaCellState(:, k) =  NetParam.deltaCellStateActive(:, k) .* activationDerivativeOut(NetParam.cellStateActive(:, k), NetParam.sigm_h);
                % NetParam.deltaCellState(:, k) = NetParam.deltaCellStateActive(:, k) .* activationDerivativeIn(NetParam.cellState(:, k), NetParam.sigm_h);
                if k > 1
                    NetParam.deltaResetGateActive(:, k) = NetParam.hiddenIrregular(:, k - 1) .* (NetParam.Uc' * NetParam.deltaCellState(:, k));
                    NetParam.deltaResetGate(:, k) = NetParam.deltaResetGateActive(:, k) .* activationDerivativeOut(NetParam.resetGateActive(:, k), NetParam.sigm_g);
                    % NetParam.deltaResetGate(:, k) = NetParam.deltaResetGateActive(:, k) .* activationDerivativeIn(NetParam.resetGate(:, k), NetParam.sigm_g);
                    NetParam.deltaUpdateGateActive(:, k) = NetParam.deltaHidden(:, k) .* (NetParam.hiddenIrregular(:, k - 1) - NetParam.cellStateActive(:, k));
                else
                    NetParam.deltaUpdateGateActive(:, k) = - NetParam.deltaHidden(:, k) .* NetParam.cellStateActive(:, k);
                end
                NetParam.deltaUpdateGate(:, k) = NetParam.deltaUpdateGateActive(:, k) .* activationDerivativeOut(NetParam.updateGateActive(:, k), NetParam.sigm_g);
                % NetParam.deltaUpdateGate(:, k) = NetParam.deltaUpdateGateActive(:, k) .* activationDerivativeIn(NetParam.updateGate(:, k), NetParam.sigm_g);
                NetParam.deltaInputFilledScaled(:, k) = NetParam.Wz' * NetParam.deltaUpdateGate(:, k) + NetParam.Wr' * NetParam.deltaResetGate(:, k) + NetParam.Wc' * NetParam.deltaCellState(:, k);
                NetParam.deltaInputFilled(:, k) = NetParam.deltaInputFilledScaled(:, k) * NetParam.inputFactor(k);
                NetParam.deltaInput(:, k) = (1 + dt_x(:, k) .* NetParam.Px) .* NetParam.deltaInputFilled(:, k);
            end
        end
        
        % Gradients of the learnable parameters across all timestamps
        NetParam.dL_Wy = NetParam.dL_Wy + NetParam.deltaOutput * NetParam.hiddenIrregular';
        NetParam.dL_by = NetParam.dL_by + sum(NetParam.deltaOutput, 2);
        NetParam.dL_Wz = NetParam.dL_Wz + NetParam.deltaUpdateGate * NetParam.inputFilledScaled';
        NetParam.dL_Uz = NetParam.dL_Uz + NetParam.deltaUpdateGate(:, 2 : end) * NetParam.hiddenIrregular(:, 1 : end - 1)' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_bz = NetParam.dL_bz + sum(NetParam.deltaUpdateGate, 2);
        NetParam.dL_Wr = NetParam.dL_Wr + NetParam.deltaResetGate * NetParam.inputFilledScaled';
        NetParam.dL_Ur = NetParam.dL_Ur + NetParam.deltaResetGate(:, 2 : end) * NetParam.hiddenIrregular(:, 1 : end - 1)' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_br = NetParam.dL_br + sum(NetParam.deltaResetGate, 2);
        NetParam.dL_Wc = NetParam.dL_Wc + NetParam.deltaCellState * NetParam.inputFilledScaled';
        NetParam.dL_Uc = NetParam.dL_Uc + NetParam.deltaCellState(:, 2 : end) * (NetParam.resetGateActive(:, 2 : end) .* NetParam.hiddenIrregular(:, 1 : end - 1))' * NetParam.sequenceSize / (NetParam.sequenceSize - 1);
        NetParam.dL_bc = NetParam.dL_bc + sum(NetParam.deltaCellState, 2);
        NetParam.dL_Ph = NetParam.dL_Ph + (NetParam.deltaHiddenIrregular * diag(dt_xy - NetParam.tau)) * NetParam.hidden' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Ch = NetParam.dL_Ch + NetParam.deltaHiddenIrregular * (dt_xy - NetParam.tau)' * NetParam.sequenceSize / (sum(~(dt_xy - NetParam.tau)) + 1e-8);
        NetParam.dL_Px = NetParam.dL_Px + sum(NetParam.deltaInputFilled .* dt_x .* NetParam.input, 2) * NetParam.sequenceSize ./ (sum(dt_x ~= 0, 2) + 1e-8);
        NetParam.dL_Cx = NetParam.dL_Cx + sum(NetParam.deltaInputFilled .* dt_x, 2) * NetParam.sequenceSize ./ (sum(dt_x ~= 0, 2) + 1e-8);
        
end

end
