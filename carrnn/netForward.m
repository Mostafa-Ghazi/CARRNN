function NetParam = netForward(NetParam, x, dt_x, dt_xy)

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

% Feedforward pass of the input through the network

% NetParam: structure including the network and optimization parameters, memory blocks, and gradients
% x: input array [inputSize x sequenceSize]
% dt_x: time intervals of the input features [inputSize x sequenceSize]
% dt_xy: time intervals of the aligned visits [1 x sequenceSize]

% Handling missing values in the input array based on the number of feature nodes per timestamp
NetParam.inputFactor = (NetParam.inputSize - sum(isnan(x), 1)) / NetParam.inputSize; % input normalization factor
NetParam.input = x; NetParam.input(isnan(x)) = 0; % missing values replaced with zeros

switch NetParam.netType
    
    case 'rnn'
        
        % Initialization of the memory blocks before and after activation
        NetParam.inputFilled = zeros(NetParam.inputSize, NetParam.sequenceSize); % imputed input array
        NetParam.inputFilledScaled = zeros(NetParam.inputSize, NetParam.sequenceSize); % scaled imputed input array
        NetParam.hidden = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % hidden/recurrent array [h(t)]
        NetParam.hiddenActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired hidden/recurrent array [sigm_h(h(t))]
        NetParam.hiddenIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % irregular hidden/recurrent array
        NetParam.output = zeros(NetParam.outputSize, NetParam.sequenceSize); % output array [y(t)]
        NetParam.outputActive = zeros(NetParam.outputSize, NetParam.sequenceSize); % output array [sigm_y(y(t))]
        
        % Calculations of the memory blocks before and after activation in the first timestamp
        NetParam.inputFilled(:, 1) = (1 + dt_x(:, 1) .* NetParam.Px) .* NetParam.input(:, 1) + dt_x(:, 1) .* NetParam.Cx;
        NetParam.inputFilledScaled(:, 1) = NetParam.inputFilled(:, 1) * NetParam.inputFactor(1);
        NetParam.hidden(:, 1) = NetParam.Wh * NetParam.inputFilledScaled(:, 1) + NetParam.bh;
        NetParam.hiddenActive(:, 1) = activationFunction(NetParam.hidden(:, 1), NetParam.sigm_h);
        NetParam.hiddenIrregular(:, 1) = (eye(NetParam.hiddenSize) + (dt_xy(1) - NetParam.tau) * NetParam.Ph) * NetParam.hiddenActive(:, 1) + (dt_xy(1) - NetParam.tau) * NetParam.Ch;
        NetParam.output(:, 1) = NetParam.Wy * NetParam.hiddenIrregular(:, 1) + NetParam.by;
        NetParam.outputActive(:, 1) = activationFunction(NetParam.output(:, 1), NetParam.sigm_y);
        
        % Feedforward through time
        for k = 2 : NetParam.sequenceSize
            NetParam.inputFilled(:, k) = (1 + dt_x(:, k) .* NetParam.Px) .* NetParam.input(:, k) + dt_x(:, k) .* NetParam.Cx;
            NetParam.inputFilledScaled(:, k) = NetParam.inputFilled(:, k) * NetParam.inputFactor(k);
            NetParam.hidden(:, k) = NetParam.Wh * NetParam.inputFilledScaled(:, 1) + NetParam.Uh * NetParam.hiddenIrregular(:, k - 1) + NetParam.bh;
            NetParam.hiddenActive(:, k) = activationFunction(NetParam.hidden(:, k), NetParam.sigm_h);
            NetParam.hiddenIrregular(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Ph) * NetParam.hiddenActive(:, k) + (dt_xy(k) - NetParam.tau) * NetParam.Ch;
            NetParam.output(:, k) = NetParam.Wy * NetParam.hiddenIrregular(:, k) + NetParam.by;
            NetParam.outputActive(:, k) = activationFunction(NetParam.output(:, k), NetParam.sigm_y);
        end
        
    case 'lstm'
        
        % Initialization of the memory blocks before and after activation
        NetParam.inputFilled = zeros(NetParam.inputSize, NetParam.sequenceSize); % imputed input array
        NetParam.inputFilledScaled = zeros(NetParam.inputSize, NetParam.sequenceSize); % scaled imputed input array
        NetParam.forgetGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % forget gate [f(t)]
        NetParam.forgetGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired forget gate [sigm_g(f(t))]
        NetParam.inputGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % input gate [i(t)]
        NetParam.inputGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired input gate [sigm_g(i(t))]
        NetParam.modulationGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % modulation gate [z(t)]
        NetParam.modulationGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired modulation gate [sigm_c(z(t))]
        NetParam.cellState = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % cell/candidate state [c(t)]
        NetParam.cellStateIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % irregular cell/candidate state
        NetParam.cellStateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired cell/candidate state [sigm_h(c(t))]
        NetParam.outputGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % output gate [o(t)]
        NetParam.outputGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired output gate [sigm_g(o(t))]
        NetParam.hidden = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % hidden/recurrent array [h(t)]
        NetParam.hiddenIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % irregular hidden/recurrent array
        NetParam.output = zeros(NetParam.outputSize, NetParam.sequenceSize); % output array [y(t)]
        NetParam.outputActive = zeros(NetParam.outputSize, NetParam.sequenceSize); % output array [sigm_y(y(t))]
        
        % Calculations of the memory blocks before and after activation in the first timestamp
        NetParam.inputFilled(:, 1) = (1 + dt_x(:, 1) .* NetParam.Px) .* NetParam.input(:, 1) + dt_x(:, 1) .* NetParam.Cx;
        NetParam.inputFilledScaled(:, 1) = NetParam.inputFilled(:, 1) * NetParam.inputFactor(1);
        NetParam.forgetGate(:, 1) = NetParam.Wf * NetParam.inputFilledScaled(:, 1) + NetParam.bf;
        NetParam.forgetGateActive(:, 1) = activationFunction(NetParam.forgetGate(:, 1), NetParam.sigm_g);
        NetParam.inputGate(:, 1) = NetParam.Wi * NetParam.inputFilledScaled(:, 1) + NetParam.bi;
        NetParam.inputGateActive(:, 1) = activationFunction(NetParam.inputGate(:, 1), NetParam.sigm_g);
        NetParam.modulationGate(:, 1) = NetParam.Wz * NetParam.inputFilledScaled(:, 1) + NetParam.bz;
        NetParam.modulationGateActive(:, 1) = activationFunction(NetParam.modulationGate(:, 1), NetParam.sigm_c);
        NetParam.cellState(:, 1) = NetParam.inputGateActive(:, 1) .* NetParam.modulationGateActive(:, 1);
        NetParam.cellStateIrregular(:, 1) = (eye(NetParam.hiddenSize) + (dt_xy(1) - NetParam.tau) * NetParam.Pc) * NetParam.cellState(:, 1) + (dt_xy(1) - NetParam.tau) * NetParam.Cc;
        NetParam.cellStateActive(:, 1) = activationFunction(NetParam.cellState(:, 1), NetParam.sigm_h);
        NetParam.outputGate(:, 1) = NetParam.Wo * NetParam.inputFilledScaled(:, 1) + NetParam.Vo * NetParam.cellStateIrregular(:, 1) + NetParam.bo;
        NetParam.outputGateActive(:, 1) = activationFunction(NetParam.outputGate(:, 1), NetParam.sigm_g);
        NetParam.hidden(:, 1) = NetParam.outputGateActive(:, 1) .* NetParam.cellStateActive(:, 1);
        NetParam.hiddenIrregular(:, 1) = (eye(NetParam.hiddenSize) + (dt_xy(1) - NetParam.tau) * NetParam.Ph) * NetParam.hidden(:, 1) + (dt_xy(1) - NetParam.tau) * NetParam.Ch;
        NetParam.output(:, 1) = NetParam.Wy * NetParam.hiddenIrregular(:, 1) + NetParam.by;
        NetParam.outputActive(:, 1) = activationFunction(NetParam.output(:, 1), NetParam.sigm_y);
        
        % Feedforward through time
        for k = 2 : NetParam.sequenceSize
            NetParam.inputFilled(:, k) = (1 + dt_x(:, k) .* NetParam.Px) .* NetParam.input(:, k) + dt_x(:, k) .* NetParam.Cx;
            NetParam.inputFilledScaled(:, k) = NetParam.inputFilled(:, k) * NetParam.inputFactor(k);
            NetParam.forgetGate(:, k) = NetParam.Wf * NetParam.inputFilledScaled(:, k) + NetParam.Uf * NetParam.hiddenIrregular(:, k - 1) + NetParam.Vf * NetParam.cellStateIrregular(:, k - 1) + NetParam.bf;
            NetParam.forgetGateActive(:, k) = activationFunction(NetParam.forgetGate(:, k), NetParam.sigm_g);
            NetParam.inputGate(:, k) = NetParam.Wi * NetParam.inputFilledScaled(:, k) + NetParam.Ui * NetParam.hiddenIrregular(:, k - 1) + NetParam.Vi * NetParam.cellStateIrregular(:, k - 1) + NetParam.bi;
            NetParam.inputGateActive(:, k) = activationFunction(NetParam.inputGate(:, k), NetParam.sigm_g);
            NetParam.modulationGate(:, k) = NetParam.Wz * NetParam.inputFilledScaled(:, k) + NetParam.Uz * NetParam.hiddenIrregular(:, k - 1) + NetParam.bz;
            NetParam.modulationGateActive(:, k) = activationFunction(NetParam.modulationGate(:, k), NetParam.sigm_c);
            NetParam.cellState(:, k) = NetParam.forgetGateActive(:, k) .* NetParam.cellStateIrregular(:, k - 1) + NetParam.inputGateActive(:, k) .* NetParam.modulationGateActive(:, k);
            NetParam.cellStateIrregular(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Pc) * NetParam.cellState(:, k) + (dt_xy(k) - NetParam.tau) * NetParam.Cc;
            NetParam.cellStateActive(:, k) = activationFunction(NetParam.cellState(:, k), NetParam.sigm_h);
            NetParam.outputGate(:, k) = NetParam.Wo * NetParam.inputFilledScaled(:, k) + NetParam.Uo * NetParam.hiddenIrregular(:, k - 1) + NetParam.Vo * NetParam.cellStateIrregular(:, k) + NetParam.bo;
            NetParam.outputGateActive(:, k) = activationFunction(NetParam.outputGate(:, k), NetParam.sigm_g);
            NetParam.hidden(:, k) = NetParam.outputGateActive(:, k) .* NetParam.cellStateActive(:, k);
            NetParam.hiddenIrregular(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Ph) * NetParam.hidden(:, k) + (dt_xy(k) - NetParam.tau) * NetParam.Ch;
            NetParam.output(:, k) = NetParam.Wy * NetParam.hiddenIrregular(:, k) + NetParam.by;
            NetParam.outputActive(:, k) = activationFunction(NetParam.output(:, k), NetParam.sigm_y);
        end
        
    case 'gru'
        
        % Initialization of the memory blocks before and after activation
        NetParam.inputFilled = zeros(NetParam.inputSize, NetParam.sequenceSize); % imputed input array
        NetParam.inputFilledScaled = zeros(NetParam.inputSize, NetParam.sequenceSize); % scaled imputed input array
        NetParam.updateGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % update gate [z(t)]
        NetParam.updateGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired update gate [sigm_g(z(t))]
        NetParam.resetGate = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % reset gate [r(t)]
        NetParam.resetGateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired reset gate [sigm_g(r(t))]
        NetParam.cellState = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % cell/candidate state [c(t)]
        NetParam.cellStateActive = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % fired cell/candidate state [sigm_h(c(t))]
        NetParam.hidden = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % hidden/recurrent array [h(t)]
        NetParam.hiddenIrregular = zeros(NetParam.hiddenSize, NetParam.sequenceSize); % irregular hidden/recurrent array
        NetParam.output = zeros(NetParam.outputSize, NetParam.sequenceSize); % output array [y(t)]
        NetParam.outputActive = zeros(NetParam.outputSize, NetParam.sequenceSize); % output array [sigm_y(y(t))]
        
        % Calculations of the memory blocks before and after activation in the first timestamp
        NetParam.inputFilled(:, 1) = (1 + dt_x(:, 1) .* NetParam.Px) .* NetParam.input(:, 1) + dt_x(:, 1) .* NetParam.Cx;
        NetParam.inputFilledScaled(:, 1) = NetParam.inputFilled(:, 1) * NetParam.inputFactor(1);
        NetParam.updateGate(:, 1) = NetParam.Wz * NetParam.inputFilledScaled(:, 1) + NetParam.bz;
        NetParam.updateGateActive(:, 1) = activationFunction(NetParam.updateGate(:, 1), NetParam.sigm_g);
        NetParam.resetGate(:, 1) = NetParam.Wr * NetParam.inputFilledScaled(:, 1) + NetParam.br;
        NetParam.resetGateActive(:, 1) = activationFunction(NetParam.resetGate(:, 1), NetParam.sigm_g);
        NetParam.cellState(:, 1) = NetParam.Wc * NetParam.inputFilledScaled(:, 1) + NetParam.bc;
        NetParam.cellStateActive(:, 1) = activationFunction(NetParam.cellState(:, 1), NetParam.sigm_h);
        NetParam.hidden(:, 1) = (1 - NetParam.updateGateActive(:, 1)) .* NetParam.cellStateActive(:, 1);
        NetParam.hiddenIrregular(:, 1) = (eye(NetParam.hiddenSize) + (dt_xy(1) - NetParam.tau) * NetParam.Ph) * NetParam.hidden(:, 1) + (dt_xy(1) - NetParam.tau) * NetParam.Ch;
        NetParam.output(:, 1) = NetParam.Wy * NetParam.hiddenIrregular(:, 1) + NetParam.by;
        NetParam.outputActive(:, 1) = activationFunction(NetParam.output(:, 1), NetParam.sigm_y);
        
        % Feedforward through time
        for k = 2 : NetParam.sequenceSize
            NetParam.inputFilled(:, k) = (1 + dt_x(:, k) .* NetParam.Px) .* NetParam.input(:, k) + dt_x(:, k) .* NetParam.Cx;
            NetParam.inputFilledScaled(:, k) = NetParam.inputFilled(:, k) * NetParam.inputFactor(k);
            NetParam.updateGate(:, k) = NetParam.Wz * NetParam.inputFilledScaled(:, k) + NetParam.Uz * NetParam.hiddenIrregular(:, k - 1) + NetParam.bz;
            NetParam.updateGateActive(:, k) = activationFunction(NetParam.updateGate(:, k), NetParam.sigm_g);
            NetParam.resetGate(:, k) = NetParam.Wr * NetParam.inputFilledScaled(:, k) + NetParam.Ur * NetParam.hiddenIrregular(:, k - 1) + NetParam.br;
            NetParam.resetGateActive(:, k) = activationFunction(NetParam.resetGate(:, k), NetParam.sigm_g);
            NetParam.cellState(:, k) = NetParam.Wc * NetParam.inputFilledScaled(:, k) + NetParam.Uc * (NetParam.resetGateActive(:, k) .* NetParam.hiddenIrregular(:, k - 1)) + NetParam.bc;
            NetParam.cellStateActive(:, k) = activationFunction(NetParam.cellState(:, k), NetParam.sigm_h);
            NetParam.hidden(:, k) = (1 - NetParam.updateGateActive(:, k)) .* NetParam.cellStateActive(:, k) + NetParam.updateGateActive(:, k) .* NetParam.hiddenIrregular(:, k - 1);
            NetParam.hiddenIrregular(:, k) = (eye(NetParam.hiddenSize) + (dt_xy(k) - NetParam.tau) * NetParam.Ph) * NetParam.hidden(:, k) + (dt_xy(k) - NetParam.tau) * NetParam.Ch;
            NetParam.output(:, k) = NetParam.Wy * NetParam.hiddenIrregular(:, k) + NetParam.by;
            NetParam.outputActive(:, k) = activationFunction(NetParam.output(:, k), NetParam.sigm_y);
        end
        
end

end
