function NetParam = netUpdate(NetParam, variableType)

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

% Updating the network parameters

% NetParam: structure including the network and optimization parameters, memory blocks, and gradients

switch variableType
    
    case 'adam'
        
        switch NetParam.netType
            
            case 'rnn'
                
                % Updating the parameters using the adaptive moment estimation method
                [NetParam.Wy, NetParam.dWy_avg, NetParam.dWy_avg_sq] = adamupdate(NetParam.Wy, NetParam.dL_Wy, NetParam.dWy_avg, NetParam.dWy_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.by, NetParam.dby_avg, NetParam.dby_avg_sq] = adamupdate(NetParam.by, NetParam.dL_by, NetParam.dby_avg, NetParam.dby_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wh, NetParam.dWh_avg, NetParam.dWh_avg_sq] = adamupdate(NetParam.Wh, NetParam.dL_Wh, NetParam.dWh_avg, NetParam.dWh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Uh, NetParam.dUh_avg, NetParam.dUh_avg_sq] = adamupdate(NetParam.Uh, NetParam.dL_Uh, NetParam.dUh_avg, NetParam.dUh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.bh, NetParam.dbh_avg, NetParam.dbh_avg_sq] = adamupdate(NetParam.bh, NetParam.dL_bh, NetParam.dbh_avg, NetParam.dbh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ph, NetParam.dPh_avg, NetParam.dPh_avg_sq] = adamupdate(NetParam.Ph, NetParam.dL_Ph, NetParam.dPh_avg, NetParam.dPh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ch, NetParam.dCh_avg, NetParam.dCh_avg_sq] = adamupdate(NetParam.Ch, NetParam.dL_Ch, NetParam.dCh_avg, NetParam.dCh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Px, NetParam.dPx_avg, NetParam.dPx_avg_sq] = adamupdate(NetParam.Px, NetParam.dL_Px, NetParam.dPx_avg, NetParam.dPx_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Cx, NetParam.dCx_avg, NetParam.dCx_avg_sq] = adamupdate(NetParam.Cx, NetParam.dL_Cx, NetParam.dCx_avg, NetParam.dCx_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                
            case 'lstm'
                
                % Updating the parameters using the adaptive moment estimation method
                [NetParam.Wy, NetParam.dWy_avg, NetParam.dWy_avg_sq] = adamupdate(NetParam.Wy, NetParam.dL_Wy, NetParam.dWy_avg, NetParam.dWy_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.by, NetParam.dby_avg, NetParam.dby_avg_sq] = adamupdate(NetParam.by, NetParam.dL_by, NetParam.dby_avg, NetParam.dby_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wf, NetParam.dWf_avg, NetParam.dWf_avg_sq] = adamupdate(NetParam.Wf, NetParam.dL_Wf, NetParam.dWf_avg, NetParam.dWf_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Uf, NetParam.dUf_avg, NetParam.dUf_avg_sq] = adamupdate(NetParam.Uf, NetParam.dL_Uf, NetParam.dUf_avg, NetParam.dUf_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Vf, NetParam.dVf_avg, NetParam.dVf_avg_sq] = adamupdate(NetParam.Vf, NetParam.dL_Vf, NetParam.dVf_avg, NetParam.dVf_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.bf, NetParam.dbf_avg, NetParam.dbf_avg_sq] = adamupdate(NetParam.bf, NetParam.dL_bf, NetParam.dbf_avg, NetParam.dbf_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wi, NetParam.dWi_avg, NetParam.dWi_avg_sq] = adamupdate(NetParam.Wi, NetParam.dL_Wi, NetParam.dWi_avg, NetParam.dWi_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ui, NetParam.dUi_avg, NetParam.dUi_avg_sq] = adamupdate(NetParam.Ui, NetParam.dL_Ui, NetParam.dUi_avg, NetParam.dUi_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Vi, NetParam.dVi_avg, NetParam.dVi_avg_sq] = adamupdate(NetParam.Vi, NetParam.dL_Vi, NetParam.dVi_avg, NetParam.dVi_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.bi, NetParam.dbi_avg, NetParam.dbi_avg_sq] = adamupdate(NetParam.bi, NetParam.dL_bi, NetParam.dbi_avg, NetParam.dbi_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wz, NetParam.dWz_avg, NetParam.dWz_avg_sq] = adamupdate(NetParam.Wz, NetParam.dL_Wz, NetParam.dWz_avg, NetParam.dWz_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Uz, NetParam.dUz_avg, NetParam.dUz_avg_sq] = adamupdate(NetParam.Uz, NetParam.dL_Uz, NetParam.dUz_avg, NetParam.dUz_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.bz, NetParam.dbz_avg, NetParam.dbz_avg_sq] = adamupdate(NetParam.bz, NetParam.dL_bz, NetParam.dbz_avg, NetParam.dbz_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wo, NetParam.dWo_avg, NetParam.dWo_avg_sq] = adamupdate(NetParam.Wo, NetParam.dL_Wo, NetParam.dWo_avg, NetParam.dWo_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Uo, NetParam.dUo_avg, NetParam.dUo_avg_sq] = adamupdate(NetParam.Uo, NetParam.dL_Uo, NetParam.dUo_avg, NetParam.dUo_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Vo, NetParam.dVo_avg, NetParam.dVo_avg_sq] = adamupdate(NetParam.Vo, NetParam.dL_Vo, NetParam.dVo_avg, NetParam.dVo_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.bo, NetParam.dbo_avg, NetParam.dbo_avg_sq] = adamupdate(NetParam.bo, NetParam.dL_bo, NetParam.dbo_avg, NetParam.dbo_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ph, NetParam.dPh_avg, NetParam.dPh_avg_sq] = adamupdate(NetParam.Ph, NetParam.dL_Ph, NetParam.dPh_avg, NetParam.dPh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ch, NetParam.dCh_avg, NetParam.dCh_avg_sq] = adamupdate(NetParam.Ch, NetParam.dL_Ch, NetParam.dCh_avg, NetParam.dCh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Pc, NetParam.dPc_avg, NetParam.dPc_avg_sq] = adamupdate(NetParam.Pc, NetParam.dL_Pc, NetParam.dPc_avg, NetParam.dPc_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Cc, NetParam.dCc_avg, NetParam.dCc_avg_sq] = adamupdate(NetParam.Cc, NetParam.dL_Cc, NetParam.dCc_avg, NetParam.dCc_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Px, NetParam.dPx_avg, NetParam.dPx_avg_sq] = adamupdate(NetParam.Px, NetParam.dL_Px, NetParam.dPx_avg, NetParam.dPx_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Cx, NetParam.dCx_avg, NetParam.dCx_avg_sq] = adamupdate(NetParam.Cx, NetParam.dL_Cx, NetParam.dCx_avg, NetParam.dCx_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                
            case 'gru'
                
                % Updating the parameters using the adaptive moment estimation method
                [NetParam.Wy, NetParam.dWy_avg, NetParam.dWy_avg_sq] = adamupdate(NetParam.Wy, NetParam.dL_Wy, NetParam.dWy_avg, NetParam.dWy_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.by, NetParam.dby_avg, NetParam.dby_avg_sq] = adamupdate(NetParam.by, NetParam.dL_by, NetParam.dby_avg, NetParam.dby_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wz, NetParam.dWz_avg, NetParam.dWz_avg_sq] = adamupdate(NetParam.Wz, NetParam.dL_Wz, NetParam.dWz_avg, NetParam.dWz_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Uz, NetParam.dUz_avg, NetParam.dUz_avg_sq] = adamupdate(NetParam.Uz, NetParam.dL_Uz, NetParam.dUz_avg, NetParam.dUz_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.bz, NetParam.dbz_avg, NetParam.dbz_avg_sq] = adamupdate(NetParam.bz, NetParam.dL_bz, NetParam.dbz_avg, NetParam.dbz_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wr, NetParam.dWr_avg, NetParam.dWr_avg_sq] = adamupdate(NetParam.Wr, NetParam.dL_Wr, NetParam.dWr_avg, NetParam.dWr_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ur, NetParam.dUr_avg, NetParam.dUr_avg_sq] = adamupdate(NetParam.Ur, NetParam.dL_Ur, NetParam.dUr_avg, NetParam.dUr_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.br, NetParam.dbr_avg, NetParam.dbr_avg_sq] = adamupdate(NetParam.br, NetParam.dL_br, NetParam.dbr_avg, NetParam.dbr_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Wc, NetParam.dWc_avg, NetParam.dWc_avg_sq] = adamupdate(NetParam.Wc, NetParam.dL_Wc, NetParam.dWc_avg, NetParam.dWc_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Uc, NetParam.dUc_avg, NetParam.dUc_avg_sq] = adamupdate(NetParam.Uc, NetParam.dL_Uc, NetParam.dUc_avg, NetParam.dUc_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.bc, NetParam.dbc_avg, NetParam.dbc_avg_sq] = adamupdate(NetParam.bc, NetParam.dL_bc, NetParam.dbc_avg, NetParam.dbc_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ph, NetParam.dPh_avg, NetParam.dPh_avg_sq] = adamupdate(NetParam.Ph, NetParam.dL_Ph, NetParam.dPh_avg, NetParam.dPh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Ch, NetParam.dCh_avg, NetParam.dCh_avg_sq] = adamupdate(NetParam.Ch, NetParam.dL_Ch, NetParam.dCh_avg, NetParam.dCh_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Px, NetParam.dPx_avg, NetParam.dPx_avg_sq] = adamupdate(NetParam.Px, NetParam.dL_Px, NetParam.dPx_avg, NetParam.dPx_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                [NetParam.Cx, NetParam.dCx_avg, NetParam.dCx_avg_sq] = adamupdate(NetParam.Cx, NetParam.dL_Cx, NetParam.dCx_avg, NetParam.dCx_avg_sq, ...
                    NetParam.iteration, NetParam.learnRate, NetParam.gradientDecayFactor, NetParam.squaredGradientDecayFactor);
                
        end
        
    case 'momentum'
        
        switch NetParam.netType
            
            case 'rnn'
                
                % Momentum gradient descent method
                NetParam.dWy = NetParam.momentum * NetParam.dWy - NetParam.learnRate * NetParam.dL_Wy;
                NetParam.dby = NetParam.momentum * NetParam.dby - NetParam.learnRate * NetParam.dL_by;
                NetParam.dWh = NetParam.momentum * NetParam.dWh - NetParam.learnRate * NetParam.dL_Wh;
                NetParam.dUh = NetParam.momentum * NetParam.dUh - NetParam.learnRate * NetParam.dL_Uh;
                NetParam.dbh = NetParam.momentum * NetParam.dbh - NetParam.learnRate * NetParam.dL_bh;
                NetParam.dPh = NetParam.momentum * NetParam.dPh - NetParam.learnRate * NetParam.dL_Ph;
                NetParam.dCh = NetParam.momentum * NetParam.dCh - NetParam.learnRate * NetParam.dL_Ch;
                NetParam.dPx = NetParam.momentum * NetParam.dPx - NetParam.learnRate * NetParam.dL_Px;
                NetParam.dCx = NetParam.momentum * NetParam.dCx - NetParam.learnRate * NetParam.dL_Cx;
                
                % Updating the parameters using the obtained velocities
                NetParam.Wy = NetParam.Wy + NetParam.dWy;
                NetParam.by = NetParam.by + NetParam.dby;
                NetParam.Wh = NetParam.Wh + NetParam.dWh;
                NetParam.Uh = NetParam.Uh + NetParam.dUh;
                NetParam.bh = NetParam.bh + NetParam.dbh;
                NetParam.Ph = NetParam.Ph + NetParam.dPh;
                NetParam.Ch = NetParam.Ch + NetParam.dCh;
                NetParam.Px = NetParam.Px + NetParam.dPx;
                NetParam.Cx = NetParam.Cx + NetParam.dCx;
                
            case 'lstm'
                
                % Momentum gradient descent method
                NetParam.dWy = NetParam.momentum * NetParam.dWy - NetParam.learnRate * NetParam.dL_Wy;
                NetParam.dby = NetParam.momentum * NetParam.dby - NetParam.learnRate * NetParam.dL_by;
                NetParam.dWf = NetParam.momentum * NetParam.dWf - NetParam.learnRate * NetParam.dL_Wf;
                NetParam.dUf = NetParam.momentum * NetParam.dUf - NetParam.learnRate * NetParam.dL_Uf;
                NetParam.dVf = NetParam.momentum * NetParam.dVf - NetParam.learnRate * NetParam.dL_Vf;
                NetParam.dbf = NetParam.momentum * NetParam.dbf - NetParam.learnRate * NetParam.dL_bf;
                NetParam.dWi = NetParam.momentum * NetParam.dWi - NetParam.learnRate * NetParam.dL_Wi;
                NetParam.dUi = NetParam.momentum * NetParam.dUi - NetParam.learnRate * NetParam.dL_Ui;
                NetParam.dVi = NetParam.momentum * NetParam.dVi - NetParam.learnRate * NetParam.dL_Vi;
                NetParam.dbi = NetParam.momentum * NetParam.dbi - NetParam.learnRate * NetParam.dL_bi;
                NetParam.dWz = NetParam.momentum * NetParam.dWz - NetParam.learnRate * NetParam.dL_Wz;
                NetParam.dUz = NetParam.momentum * NetParam.dUz - NetParam.learnRate * NetParam.dL_Uz;
                NetParam.dbz = NetParam.momentum * NetParam.dbz - NetParam.learnRate * NetParam.dL_bz;
                NetParam.dWo = NetParam.momentum * NetParam.dWo - NetParam.learnRate * NetParam.dL_Wo;
                NetParam.dUo = NetParam.momentum * NetParam.dUo - NetParam.learnRate * NetParam.dL_Uo;
                NetParam.dVo = NetParam.momentum * NetParam.dVo - NetParam.learnRate * NetParam.dL_Vo;
                NetParam.dbo = NetParam.momentum * NetParam.dbo - NetParam.learnRate * NetParam.dL_bo;
                NetParam.dPh = NetParam.momentum * NetParam.dPh - NetParam.learnRate * NetParam.dL_Ph;
                NetParam.dCh = NetParam.momentum * NetParam.dCh - NetParam.learnRate * NetParam.dL_Ch;
                NetParam.dPc = NetParam.momentum * NetParam.dPc - NetParam.learnRate * NetParam.dL_Pc;
                NetParam.dCc = NetParam.momentum * NetParam.dCc - NetParam.learnRate * NetParam.dL_Cc;
                NetParam.dPx = NetParam.momentum * NetParam.dPx - NetParam.learnRate * NetParam.dL_Px;
                NetParam.dCx = NetParam.momentum * NetParam.dCx - NetParam.learnRate * NetParam.dL_Cx;
                
                % Updating the parameters using the obtained velocities
                NetParam.Wy = NetParam.Wy + NetParam.dWy;
                NetParam.by = NetParam.by + NetParam.dby;
                NetParam.Wf = NetParam.Wf + NetParam.dWf;
                NetParam.Uf = NetParam.Uf + NetParam.dUf;
                NetParam.Vf = NetParam.Vf + NetParam.dVf;
                NetParam.bf = NetParam.bf + NetParam.dbf;
                NetParam.Wi = NetParam.Wi + NetParam.dWi;
                NetParam.Ui = NetParam.Ui + NetParam.dUi;
                NetParam.Vi = NetParam.Vi + NetParam.dVi;
                NetParam.bi = NetParam.bi + NetParam.dbi;
                NetParam.Wz = NetParam.Wz + NetParam.dWz;
                NetParam.Uz = NetParam.Uz + NetParam.dUz;
                NetParam.bz = NetParam.bz + NetParam.dbz;
                NetParam.Wo = NetParam.Wo + NetParam.dWo;
                NetParam.Uo = NetParam.Uo + NetParam.dUo;
                NetParam.Vo = NetParam.Vo + NetParam.dVo;
                NetParam.bo = NetParam.bo + NetParam.dbo;
                NetParam.Ph = NetParam.Ph + NetParam.dPh;
                NetParam.Ch = NetParam.Ch + NetParam.dCh;
                NetParam.Pc = NetParam.Pc + NetParam.dPc;
                NetParam.Cc = NetParam.Cc + NetParam.dCc;
                NetParam.Px = NetParam.Px + NetParam.dPx;
                NetParam.Cx = NetParam.Cx + NetParam.dCx;
                
            case 'gru'
                
                % Momentum gradient descent method
                NetParam.dWy = NetParam.momentum * NetParam.dWy - NetParam.learnRate * NetParam.dL_Wy;
                NetParam.dby = NetParam.momentum * NetParam.dby - NetParam.learnRate * NetParam.dL_by;
                NetParam.dWz = NetParam.momentum * NetParam.dWz - NetParam.learnRate * NetParam.dL_Wz;
                NetParam.dUz = NetParam.momentum * NetParam.dUz - NetParam.learnRate * NetParam.dL_Uz;
                NetParam.dbz = NetParam.momentum * NetParam.dbz - NetParam.learnRate * NetParam.dL_bz;
                NetParam.dWr = NetParam.momentum * NetParam.dWr - NetParam.learnRate * NetParam.dL_Wr;
                NetParam.dUr = NetParam.momentum * NetParam.dUr - NetParam.learnRate * NetParam.dL_Ur;
                NetParam.dbr = NetParam.momentum * NetParam.dbr - NetParam.learnRate * NetParam.dL_br;
                NetParam.dWc = NetParam.momentum * NetParam.dWc - NetParam.learnRate * NetParam.dL_Wc;
                NetParam.dUc = NetParam.momentum * NetParam.dUc - NetParam.learnRate * NetParam.dL_Uc;
                NetParam.dbc = NetParam.momentum * NetParam.dbc - NetParam.learnRate * NetParam.dL_bc;
                NetParam.dPh = NetParam.momentum * NetParam.dPh - NetParam.learnRate * NetParam.dL_Ph;
                NetParam.dCh = NetParam.momentum * NetParam.dCh - NetParam.learnRate * NetParam.dL_Ch;
                NetParam.dPx = NetParam.momentum * NetParam.dPx - NetParam.learnRate * NetParam.dL_Px;
                NetParam.dCx = NetParam.momentum * NetParam.dCx - NetParam.learnRate * NetParam.dL_Cx;
                
                % Updating the parameters using the obtained velocities
                NetParam.Wy = NetParam.Wy + NetParam.dWy;
                NetParam.by = NetParam.by + NetParam.dby;
                NetParam.Wz = NetParam.Wz + NetParam.dWz;
                NetParam.Uz = NetParam.Uz + NetParam.dUz;
                NetParam.bz = NetParam.bz + NetParam.dbz;
                NetParam.Wr = NetParam.Wr + NetParam.dWr;
                NetParam.Ur = NetParam.Ur + NetParam.dUr;
                NetParam.br = NetParam.br + NetParam.dbr;
                NetParam.Wc = NetParam.Wc + NetParam.dWc;
                NetParam.Uc = NetParam.Uc + NetParam.dUc;
                NetParam.bc = NetParam.bc + NetParam.dbc;
                NetParam.Ph = NetParam.Ph + NetParam.dPh;
                NetParam.Ch = NetParam.Ch + NetParam.dCh;
                NetParam.Px = NetParam.Px + NetParam.dPx;
                NetParam.Cx = NetParam.Cx + NetParam.dCx;
                
        end
        
    case 'gradients'
        
        switch NetParam.netType
            case 'rnn'
                NetParam.dL_Wy = NetParam.dL_Wy / NetParam.miniBatchSize;
                NetParam.dL_by = NetParam.dL_by / NetParam.miniBatchSize;
                NetParam.dL_Wh = NetParam.dL_Wh / NetParam.miniBatchSize;
                NetParam.dL_Uh = NetParam.dL_Uh / NetParam.miniBatchSize;
                NetParam.dL_bh = NetParam.dL_bh / NetParam.miniBatchSize;
                NetParam.dL_Ph = NetParam.dL_Ph / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Ch = NetParam.dL_Ch / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Px = NetParam.dL_Px ./ (NetParam.miniBatchSizeFilled + 1e-8);
                NetParam.dL_Cx = NetParam.dL_Cx ./ (NetParam.miniBatchSizeFilled + 1e-8);
            case 'lstm'
                NetParam.dL_Wy = NetParam.dL_Wy / NetParam.miniBatchSize;
                NetParam.dL_by = NetParam.dL_by / NetParam.miniBatchSize;
                NetParam.dL_Wf = NetParam.dL_Wf / NetParam.miniBatchSize;
                NetParam.dL_Uf = NetParam.dL_Uf / NetParam.miniBatchSize;
                NetParam.dL_Vf = NetParam.dL_Vf / NetParam.miniBatchSize;
                NetParam.dL_bf = NetParam.dL_bf / NetParam.miniBatchSize;
                NetParam.dL_Wi = NetParam.dL_Wi / NetParam.miniBatchSize;
                NetParam.dL_Ui = NetParam.dL_Ui / NetParam.miniBatchSize;
                NetParam.dL_Vi = NetParam.dL_Vi / NetParam.miniBatchSize;
                NetParam.dL_bi = NetParam.dL_bi / NetParam.miniBatchSize;
                NetParam.dL_Wz = NetParam.dL_Wz / NetParam.miniBatchSize;
                NetParam.dL_Uz = NetParam.dL_Uz / NetParam.miniBatchSize;
                NetParam.dL_bz = NetParam.dL_bz / NetParam.miniBatchSize;
                NetParam.dL_Wo = NetParam.dL_Wo / NetParam.miniBatchSize;
                NetParam.dL_Uo = NetParam.dL_Uo / NetParam.miniBatchSize;
                NetParam.dL_Vo = NetParam.dL_Vo / NetParam.miniBatchSize;
                NetParam.dL_bo = NetParam.dL_bo / NetParam.miniBatchSize;
                NetParam.dL_Ph = NetParam.dL_Ph / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Ch = NetParam.dL_Ch / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Pc = NetParam.dL_Pc / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Cc = NetParam.dL_Cc / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Px = NetParam.dL_Px ./ (NetParam.miniBatchSizeFilled + 1e-8);
                NetParam.dL_Cx = NetParam.dL_Cx ./ (NetParam.miniBatchSizeFilled + 1e-8);
            case 'gru'
                NetParam.dL_Wy = NetParam.dL_Wy / NetParam.miniBatchSize;
                NetParam.dL_by = NetParam.dL_by / NetParam.miniBatchSize;
                NetParam.dL_Wz = NetParam.dL_Wz / NetParam.miniBatchSize;
                NetParam.dL_Uz = NetParam.dL_Uz / NetParam.miniBatchSize;
                NetParam.dL_bz = NetParam.dL_bz / NetParam.miniBatchSize;
                NetParam.dL_Wr = NetParam.dL_Wr / NetParam.miniBatchSize;
                NetParam.dL_Ur = NetParam.dL_Ur / NetParam.miniBatchSize;
                NetParam.dL_br = NetParam.dL_br / NetParam.miniBatchSize;
                NetParam.dL_Wc = NetParam.dL_Wc / NetParam.miniBatchSize;
                NetParam.dL_Uc = NetParam.dL_Uc / NetParam.miniBatchSize;
                NetParam.dL_bc = NetParam.dL_bc / NetParam.miniBatchSize;
                NetParam.dL_Ph = NetParam.dL_Ph / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Ch = NetParam.dL_Ch / (NetParam.miniBatchSizeIrregular + 1e-8);
                NetParam.dL_Px = NetParam.dL_Px ./ (NetParam.miniBatchSizeFilled + 1e-8);
                NetParam.dL_Cx = NetParam.dL_Cx ./ (NetParam.miniBatchSizeFilled + 1e-8);
        end
        
end

end
