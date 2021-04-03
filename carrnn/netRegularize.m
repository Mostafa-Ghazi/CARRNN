function NetParam = netRegularize(NetParam)

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

% Regularization of the network weights

% NetParam: structure including the network and optimization parameters, memory blocks, and gradients

switch NetParam.netType
    
    case 'rnn'
        
        switch NetParam.regularizationMethod
            case 'l2norm' % L2 regularization of the weights
                NetParam.dL_Wy = NetParam.dL_Wy + NetParam.weightDecay * NetParam.Wy / numel(NetParam.Wy);
                NetParam.dL_Wh = NetParam.dL_Wh + NetParam.weightDecay * NetParam.Wh / numel(NetParam.Wh);
                NetParam.dL_Uh = NetParam.dL_Uh + NetParam.weightDecay * NetParam.Uh / numel(NetParam.Uh);
                NetParam.dL_Ph = NetParam.dL_Ph + NetParam.weightDecay * NetParam.Ph / numel(NetParam.Ph);
                NetParam.dL_Px = NetParam.dL_Px + NetParam.weightDecay * NetParam.Px / numel(NetParam.Px);
            case 'l1norm' % L1 regularization of the weights
                NetParam.dL_Wy = NetParam.dL_Wy + NetParam.weightDecay * sign(NetParam.Wy) / numel(NetParam.Wy);
                NetParam.dL_Wh = NetParam.dL_Wh + NetParam.weightDecay * sign(NetParam.Wh) / numel(NetParam.Wh);
                NetParam.dL_Uh = NetParam.dL_Uh + NetParam.weightDecay * sign(NetParam.Uh) / numel(NetParam.Uh);
                NetParam.dL_Ph = NetParam.dL_Ph + NetParam.weightDecay * sign(NetParam.Ph) / numel(NetParam.Ph);
                NetParam.dL_Px = NetParam.dL_Px + NetParam.weightDecay * sign(NetParam.Px) / numel(NetParam.Px);
        end
        
    case 'lstm'
        
        switch NetParam.regularizationMethod
            case 'l2norm' % L2 regularization of the weights
                NetParam.dL_Wy = NetParam.dL_Wy + NetParam.weightDecay * NetParam.Wy / numel(NetParam.Wy);
                NetParam.dL_Wf = NetParam.dL_Wf + NetParam.weightDecay * NetParam.Wf / numel(NetParam.Wf);
                NetParam.dL_Uf = NetParam.dL_Uf + NetParam.weightDecay * NetParam.Uf / numel(NetParam.Uf);
                NetParam.dL_Vf = NetParam.dL_Vf + NetParam.weightDecay * NetParam.Vf / numel(diag(NetParam.Vf));
                NetParam.dL_Wi = NetParam.dL_Wi + NetParam.weightDecay * NetParam.Wi / numel(NetParam.Wi);
                NetParam.dL_Ui = NetParam.dL_Ui + NetParam.weightDecay * NetParam.Ui / numel(NetParam.Ui);
                NetParam.dL_Vi = NetParam.dL_Vi + NetParam.weightDecay * NetParam.Vi / numel(diag(NetParam.Vi));
                NetParam.dL_Wz = NetParam.dL_Wz + NetParam.weightDecay * NetParam.Wz / numel(NetParam.Wz);
                NetParam.dL_Uz = NetParam.dL_Uz + NetParam.weightDecay * NetParam.Uz / numel(NetParam.Uz);
                NetParam.dL_Wo = NetParam.dL_Wo + NetParam.weightDecay * NetParam.Wo / numel(NetParam.Wo);
                NetParam.dL_Uo = NetParam.dL_Uo + NetParam.weightDecay * NetParam.Uo / numel(NetParam.Uo);
                NetParam.dL_Vo = NetParam.dL_Vo + NetParam.weightDecay * NetParam.Vo / numel(diag(NetParam.Vo));
                NetParam.dL_Ph = NetParam.dL_Ph + NetParam.weightDecay * NetParam.Ph / numel(NetParam.Ph);
                NetParam.dL_Pc = NetParam.dL_Pc + NetParam.weightDecay * NetParam.Pc / numel(NetParam.Pc);
                NetParam.dL_Px = NetParam.dL_Px + NetParam.weightDecay * NetParam.Px / numel(NetParam.Px);
            case 'l1norm' % L1 regularization of the weights
                NetParam.dL_Wy = NetParam.dL_Wy + NetParam.weightDecay * sign(NetParam.Wy) / numel(NetParam.Wy);
                NetParam.dL_Wf = NetParam.dL_Wf + NetParam.weightDecay * sign(NetParam.Wf) / numel(NetParam.Wf);
                NetParam.dL_Uf = NetParam.dL_Uf + NetParam.weightDecay * sign(NetParam.Uf) / numel(NetParam.Uf);
                NetParam.dL_Vf = NetParam.dL_Vf + NetParam.weightDecay * sign(NetParam.Vf) / numel(diag(NetParam.Vf));
                NetParam.dL_Wi = NetParam.dL_Wi + NetParam.weightDecay * sign(NetParam.Wi) / numel(NetParam.Wi);
                NetParam.dL_Ui = NetParam.dL_Ui + NetParam.weightDecay * sign(NetParam.Ui) / numel(NetParam.Ui);
                NetParam.dL_Vi = NetParam.dL_Vi + NetParam.weightDecay * sign(NetParam.Vi) / numel(diag(NetParam.Vi));
                NetParam.dL_Wz = NetParam.dL_Wz + NetParam.weightDecay * sign(NetParam.Wz) / numel(NetParam.Wz);
                NetParam.dL_Uz = NetParam.dL_Uz + NetParam.weightDecay * sign(NetParam.Uz) / numel(NetParam.Uz);
                NetParam.dL_Wo = NetParam.dL_Wo + NetParam.weightDecay * sign(NetParam.Wo) / numel(NetParam.Wo);
                NetParam.dL_Uo = NetParam.dL_Uo + NetParam.weightDecay * sign(NetParam.Uo) / numel(NetParam.Uo);
                NetParam.dL_Vo = NetParam.dL_Vo + NetParam.weightDecay * sign(NetParam.Vo) / numel(diag(NetParam.Vo));
                NetParam.dL_Ph = NetParam.dL_Ph + NetParam.weightDecay * sign(NetParam.Ph) / numel(NetParam.Ph);
                NetParam.dL_Pc = NetParam.dL_Pc + NetParam.weightDecay * sign(NetParam.Pc) / numel(NetParam.Pc);
                NetParam.dL_Px = NetParam.dL_Px + NetParam.weightDecay * sign(NetParam.Px) / numel(NetParam.Px);
        end
        
    case 'gru'
        
        switch NetParam.regularizationMethod
            case 'l2norm' % L2 regularization of the weights
                NetParam.dL_Wy = NetParam.dL_Wy + NetParam.weightDecay * NetParam.Wy / numel(NetParam.Wy);
                NetParam.dL_Wz = NetParam.dL_Wz + NetParam.weightDecay * NetParam.Wz / numel(NetParam.Wz);
                NetParam.dL_Uz = NetParam.dL_Uz + NetParam.weightDecay * NetParam.Uz / numel(NetParam.Uz);
                NetParam.dL_Wr = NetParam.dL_Wr + NetParam.weightDecay * NetParam.Wr / numel(NetParam.Wr);
                NetParam.dL_Ur = NetParam.dL_Ur + NetParam.weightDecay * NetParam.Ur / numel(NetParam.Ur);
                NetParam.dL_Wc = NetParam.dL_Wc + NetParam.weightDecay * NetParam.Wc / numel(NetParam.Wc);
                NetParam.dL_Uc = NetParam.dL_Uc + NetParam.weightDecay * NetParam.Uc / numel(NetParam.Uc);
                NetParam.dL_Ph = NetParam.dL_Ph + NetParam.weightDecay * NetParam.Ph / numel(NetParam.Ph);
                NetParam.dL_Px = NetParam.dL_Px + NetParam.weightDecay * NetParam.Px / numel(NetParam.Px);
            case 'l1norm' % L1 regularization of the weights
                NetParam.dL_Wy = NetParam.dL_Wy + NetParam.weightDecay * sign(NetParam.Wy) / numel(NetParam.Wy);
                NetParam.dL_Wz = NetParam.dL_Wz + NetParam.weightDecay * sign(NetParam.Wz) / numel(NetParam.Wz);
                NetParam.dL_Uz = NetParam.dL_Uz + NetParam.weightDecay * sign(NetParam.Uz) / numel(NetParam.Uz);
                NetParam.dL_Wr = NetParam.dL_Wr + NetParam.weightDecay * sign(NetParam.Wr) / numel(NetParam.Wr);
                NetParam.dL_Ur = NetParam.dL_Ur + NetParam.weightDecay * sign(NetParam.Ur) / numel(NetParam.Ur);
                NetParam.dL_Wc = NetParam.dL_Wc + NetParam.weightDecay * sign(NetParam.Wc) / numel(NetParam.Wc);
                NetParam.dL_Uc = NetParam.dL_Uc + NetParam.weightDecay * sign(NetParam.Uc) / numel(NetParam.Uc);
                NetParam.dL_Ph = NetParam.dL_Ph + NetParam.weightDecay * sign(NetParam.Ph) / numel(NetParam.Ph);
                NetParam.dL_Px = NetParam.dL_Px + NetParam.weightDecay * sign(NetParam.Px) / numel(NetParam.Px);
        end
        
end

end