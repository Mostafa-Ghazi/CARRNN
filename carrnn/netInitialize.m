function NetParam = netInitialize(NetParam, variableType)

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

% Initialization of the network parameters

% NetParam: structure including the network and optimization parameters, memory blocks, and gradients

switch variableType
    
    case 'parameters'
        
        switch NetParam.netType
            
            case 'rnn'
                
                % Initialization of the learnable parameters
                NetParam.Wy = randn(NetParam.outputSize, NetParam.hiddenSize) * sqrt(2 / (NetParam.outputSize + NetParam.hiddenSize)); % irregular hidden - output connecting weights array
                NetParam.by = zeros(NetParam.outputSize, 1); % output bias vector
                NetParam.Wh = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - hidden connecting weights array
                NetParam.Uh = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % irregular hidden - hidden connecting weights array
                NetParam.bh = zeros(NetParam.hiddenSize, 1); % hidden bias vector
                NetParam.Ph = zeros(NetParam.hiddenSize, NetParam.hiddenSize); % hidden - irregular hidden connecting weights array
                NetParam.Ch = zeros(NetParam.hiddenSize, 1); % irregular hidden bias vector
                NetParam.Px = zeros(NetParam.inputSize, 1); % input - irregular input connecting weights vector
                NetParam.Cx = zeros(NetParam.inputSize, 1); % irregular input bias vector
                
                switch NetParam.optimType
                    
                    case 'momentum'
                        
                        % Initialization of the parameter velocities (updates)
                        NetParam.dWy = zeros(NetParam.outputSize, NetParam.hiddenSize);
                        NetParam.dby = zeros(NetParam.outputSize, 1);
                        NetParam.dWh = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUh = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbh = zeros(NetParam.hiddenSize, 1);
                        NetParam.dPh = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dCh = zeros(NetParam.hiddenSize, 1);
                        NetParam.dPx = zeros(NetParam.inputSize, 1);
                        NetParam.dCx = zeros(NetParam.inputSize, 1);
                        
                    case 'adam'
                        
                        % Initialization of the average gradients
                        NetParam.dWy_avg = [];
                        NetParam.dby_avg = [];
                        NetParam.dWh_avg = [];
                        NetParam.dUh_avg = [];
                        NetParam.dbh_avg = [];
                        NetParam.dPh_avg = [];
                        NetParam.dCh_avg = [];
                        NetParam.dPx_avg = [];
                        NetParam.dCx_avg = [];
                        
                        % Initialization of the average squared gradients
                        NetParam.dWy_avg_sq = [];
                        NetParam.dby_avg_sq = [];
                        NetParam.dWh_avg_sq = [];
                        NetParam.dUh_avg_sq = [];
                        NetParam.dbh_avg_sq = [];
                        NetParam.dPh_avg_sq = [];
                        NetParam.dCh_avg_sq = [];
                        NetParam.dPx_avg_sq = [];
                        NetParam.dCx_avg_sq = [];
                        
                end
                
            case 'lstm'
                
                % Initialization of the learnable parameters
                NetParam.Wy = randn(NetParam.outputSize, NetParam.hiddenSize) * sqrt(2 / (NetParam.outputSize + NetParam.hiddenSize)); % irregular hidden - output connecting weights array
                NetParam.by = zeros(NetParam.outputSize, 1); % output bias vector
                NetParam.Wf = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - forget gate connecting weights array
                NetParam.Uf = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % hidden - forget gate connecting weights array
                NetParam.Vf = diag(randn(NetParam.hiddenSize, 1)) * sqrt(1); % cell state - forget gate connecting weights array
                NetParam.bf = zeros(NetParam.hiddenSize, 1); % forget gate bias vector
                NetParam.Wi = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - input gate connecting weights array
                NetParam.Ui = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % hidden - input gate connecting weights array
                NetParam.Vi = diag(randn(NetParam.hiddenSize, 1)) * sqrt(1); % cell state - input gate connecting weights array
                NetParam.bi = zeros(NetParam.hiddenSize, 1); % input gate bias vector
                NetParam.Wz = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - modulation gate connecting weights array
                NetParam.Uz = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % hidden - modulation gate connecting weights array
                NetParam.bz = zeros(NetParam.hiddenSize, 1); % modulation gate bias vector
                NetParam.Wo = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - output gate connecting weights array
                NetParam.Uo = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % hidden - output gate connecting weights array
                NetParam.Vo = diag(randn(NetParam.hiddenSize, 1)) * sqrt(1); % cell state - output gate connecting weights vector
                NetParam.bo = zeros(NetParam.hiddenSize, 1); % output gate bias vector
                NetParam.Ph = zeros(NetParam.hiddenSize, NetParam.hiddenSize); % hidden - irregular hidden connecting weights array
                NetParam.Ch = zeros(NetParam.hiddenSize, 1); % irregular hidden bias vector
                NetParam.Pc = zeros(NetParam.hiddenSize, NetParam.hiddenSize); % cell state - irregular cell state connecting weights array
                NetParam.Cc = zeros(NetParam.hiddenSize, 1); % irregular cell state bias vector
                NetParam.Px = zeros(NetParam.inputSize, 1); % input - irregular input connecting weights vector
                NetParam.Cx = zeros(NetParam.inputSize, 1); % irregular input bias vector
                
                switch NetParam.optimType
                    
                    case 'momentum'
                        
                        % Initialization of the parameter velocities (updates)
                        NetParam.dWy = zeros(NetParam.outputSize, NetParam.hiddenSize);
                        NetParam.dby = zeros(NetParam.outputSize, 1);
                        NetParam.dWf = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUf = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dVf = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbf = zeros(NetParam.hiddenSize, 1);
                        NetParam.dWi = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUi = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dVi = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbi = zeros(NetParam.hiddenSize, 1);
                        NetParam.dWz = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUz = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbz = zeros(NetParam.hiddenSize, 1);
                        NetParam.dWo = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUo = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dVo = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbo = zeros(NetParam.hiddenSize, 1);
                        NetParam.dPh = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dCh = zeros(NetParam.hiddenSize, 1);
                        NetParam.dPc = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dCc = zeros(NetParam.hiddenSize, 1);
                        NetParam.dPx = zeros(NetParam.inputSize, 1);
                        NetParam.dCx = zeros(NetParam.inputSize, 1);
                        
                    case 'adam'
                        
                        % Initialization of the average gradients
                        NetParam.dWy_avg = [];
                        NetParam.dby_avg = [];
                        NetParam.dWf_avg = [];
                        NetParam.dUf_avg = [];
                        NetParam.dVf_avg = [];
                        NetParam.dbf_avg = [];
                        NetParam.dWi_avg = [];
                        NetParam.dUi_avg = [];
                        NetParam.dVi_avg = [];
                        NetParam.dbi_avg = [];
                        NetParam.dWz_avg = [];
                        NetParam.dUz_avg = [];
                        NetParam.dbz_avg = [];
                        NetParam.dWo_avg = [];
                        NetParam.dUo_avg = [];
                        NetParam.dVo_avg = [];
                        NetParam.dbo_avg = [];
                        NetParam.dPh_avg = [];
                        NetParam.dCh_avg = [];
                        NetParam.dPc_avg = [];
                        NetParam.dCc_avg = [];
                        NetParam.dPx_avg = [];
                        NetParam.dCx_avg = [];
                        
                        % Initialization of the average squared gradients
                        NetParam.dWy_avg_sq = [];
                        NetParam.dby_avg_sq = [];
                        NetParam.dWf_avg_sq = [];
                        NetParam.dUf_avg_sq = [];
                        NetParam.dVf_avg_sq = [];
                        NetParam.dbf_avg_sq = [];
                        NetParam.dWi_avg_sq = [];
                        NetParam.dUi_avg_sq = [];
                        NetParam.dVi_avg_sq = [];
                        NetParam.dbi_avg_sq = [];
                        NetParam.dWz_avg_sq = [];
                        NetParam.dUz_avg_sq = [];
                        NetParam.dbz_avg_sq = [];
                        NetParam.dWo_avg_sq = [];
                        NetParam.dUo_avg_sq = [];
                        NetParam.dVo_avg_sq = [];
                        NetParam.dbo_avg_sq = [];
                        NetParam.dPh_avg_sq = [];
                        NetParam.dCh_avg_sq = [];
                        NetParam.dPc_avg_sq = [];
                        NetParam.dCc_avg_sq = [];
                        NetParam.dPx_avg_sq = [];
                        NetParam.dCx_avg_sq = [];
                        
                end
                
            case 'gru'
                
                % Initialization of the learnable parameters
                NetParam.Wy = randn(NetParam.outputSize, NetParam.hiddenSize) * sqrt(2 / (NetParam.outputSize + NetParam.hiddenSize)); % irregular hidden - output connecting weights array
                NetParam.by = zeros(NetParam.outputSize, 1); % output bias vector
                NetParam.Wz = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - update gate connecting weights array
                NetParam.Uz = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % hidden - update gate connecting weights array
                NetParam.bz = zeros(NetParam.hiddenSize, 1); % update gate bias vector
                NetParam.Wr = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - reset gate connecting weights array
                NetParam.Ur = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % hidden - reset gate connecting weights array
                NetParam.br = zeros(NetParam.hiddenSize, 1); % reset gate bias vector
                NetParam.Wc = randn(NetParam.hiddenSize, NetParam.inputSize) * sqrt(2 / (NetParam.inputSize + NetParam.hiddenSize)); % irregular input - cell state connecting weights array
                NetParam.Uc = randn(NetParam.hiddenSize, NetParam.hiddenSize) * sqrt(1 / NetParam.hiddenSize); % hidden - cell state connecting weights array
                NetParam.bc = zeros(NetParam.hiddenSize, 1); % cell state bias vector
                NetParam.Ph = zeros(NetParam.hiddenSize, NetParam.hiddenSize); % hidden - irregular hidden connecting weights array
                NetParam.Ch = zeros(NetParam.hiddenSize, 1); % irregular hidden bias vector
                NetParam.Px = zeros(NetParam.inputSize, 1); % input - irregular input connecting weights vector
                NetParam.Cx = zeros(NetParam.inputSize, 1); % irregular input bias vector
                
                switch NetParam.optimType
                    
                    case 'momentum'
                        
                        % Initialization of the parameter velocities (updates)
                        NetParam.dWy = zeros(NetParam.outputSize, NetParam.hiddenSize);
                        NetParam.dby = zeros(NetParam.outputSize, 1);
                        NetParam.dWz = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUz = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbz = zeros(NetParam.hiddenSize, 1);
                        NetParam.dWr = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUr = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbr = zeros(NetParam.hiddenSize, 1);
                        NetParam.dWc = zeros(NetParam.hiddenSize, NetParam.inputSize);
                        NetParam.dUc = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dbc = zeros(NetParam.hiddenSize, 1);
                        NetParam.dPh = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                        NetParam.dCh = zeros(NetParam.hiddenSize, 1);
                        NetParam.dPx = zeros(NetParam.inputSize, 1);
                        NetParam.dCx = zeros(NetParam.inputSize, 1);
                        
                    case 'adam'
                        
                        % Initialization of the average gradients
                        NetParam.dWy_avg = [];
                        NetParam.dby_avg = [];
                        NetParam.dWz_avg = [];
                        NetParam.dUz_avg = [];
                        NetParam.dbz_avg = [];
                        NetParam.dWr_avg = [];
                        NetParam.dUr_avg = [];
                        NetParam.dbr_avg = [];
                        NetParam.dWc_avg = [];
                        NetParam.dUc_avg = [];
                        NetParam.dbc_avg = [];
                        NetParam.dPh_avg = [];
                        NetParam.dCh_avg = [];
                        NetParam.dPx_avg = [];
                        NetParam.dCx_avg = [];
                        
                        % Initialization of the average squared gradients
                        NetParam.dWy_avg_sq = [];
                        NetParam.dby_avg_sq = [];
                        NetParam.dWz_avg_sq = [];
                        NetParam.dUz_avg_sq = [];
                        NetParam.dbz_avg_sq = [];
                        NetParam.dWr_avg_sq = [];
                        NetParam.dUr_avg_sq = [];
                        NetParam.dbr_avg_sq = [];
                        NetParam.dWc_avg_sq = [];
                        NetParam.dUc_avg_sq = [];
                        NetParam.dbc_avg_sq = [];
                        NetParam.dPh_avg_sq = [];
                        NetParam.dCh_avg_sq = [];
                        NetParam.dPx_avg_sq = [];
                        NetParam.dCx_avg_sq = [];
                        
                end
                
        end
        
    case 'gradients'
        
        switch NetParam.netType
            case 'rnn'
                NetParam.dL_Wy = zeros(NetParam.outputSize, NetParam.hiddenSize);
                NetParam.dL_by = zeros(NetParam.outputSize, 1);
                NetParam.dL_Wh = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Uh = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_bh = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Ph = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_Ch = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Px = zeros(NetParam.inputSize, 1);
                NetParam.dL_Cx = zeros(NetParam.inputSize, 1);
            case 'lstm'
                NetParam.dL_Wy = zeros(NetParam.outputSize, NetParam.hiddenSize);
                NetParam.dL_by = zeros(NetParam.outputSize, 1);
                NetParam.dL_Wf = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Uf = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_Vf = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_bf = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Wi = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Ui = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_Vi = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_bi = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Wz = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Uz = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_bz = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Wo = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Uo = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_Vo = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_bo = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Ph = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_Ch = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Pc = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_Cc = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Px = zeros(NetParam.inputSize, 1);
                NetParam.dL_Cx = zeros(NetParam.inputSize, 1);
            case 'gru'
                NetParam.dL_Wy = zeros(NetParam.outputSize, NetParam.hiddenSize);
                NetParam.dL_by = zeros(NetParam.outputSize, 1);
                NetParam.dL_Wz = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Uz = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_bz = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Wr = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Ur = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_br = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Wc = zeros(NetParam.hiddenSize, NetParam.inputSize);
                NetParam.dL_Uc = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_bc = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Ph = zeros(NetParam.hiddenSize, NetParam.hiddenSize);
                NetParam.dL_Ch = zeros(NetParam.hiddenSize, 1);
                NetParam.dL_Px = zeros(NetParam.inputSize, 1);
                NetParam.dL_Cx = zeros(NetParam.inputSize, 1);
        end
        
end

end