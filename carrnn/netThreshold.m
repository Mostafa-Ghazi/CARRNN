function NetParam = netThreshold(NetParam)

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

% Thresholding the network gradients

% NetParam: structure including the network and optimization parameters, memory blocks, and gradients

switch NetParam.netType
    
    case 'rnn'
        
        switch NetParam.gradientThresholdMethod
            case 'absolute-value' % clip the gradients exceeding the threshold
                NetParam.dL_Wy(NetParam.dL_Wy > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wy(NetParam.dL_Wy < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_by(NetParam.dL_by > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_by(NetParam.dL_by < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wh(NetParam.dL_Wh > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wh(NetParam.dL_Wh < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Uh(NetParam.dL_Uh > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Uh(NetParam.dL_Uh < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_bh(NetParam.dL_bh > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_bh(NetParam.dL_bh < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ph(NetParam.dL_Ph > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ph(NetParam.dL_Ph < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ch(NetParam.dL_Ch > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ch(NetParam.dL_Ch < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Px(NetParam.dL_Px > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Px(NetParam.dL_Px < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Cx(NetParam.dL_Cx > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Cx(NetParam.dL_Cx < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
            case 'l2norm' % clip the gradients whose norm exceed the threshold
                gradientNorm = sqrt(sum(NetParam.dL_Wy(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wy = NetParam.dL_Wy * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_by(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_by = NetParam.dL_by * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wh(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wh = NetParam.dL_Wh * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Uh(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Uh = NetParam.dL_Uh * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_bh(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_bh = NetParam.dL_bh * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ph(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ph = NetParam.dL_Ph * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ch(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ch = NetParam.dL_Ch * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Px(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Px = NetParam.dL_Px * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Cx(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Cx = NetParam.dL_Cx * (NetParam.gradientThreshold / gradientNorm);
                end
        end
        
    case 'lstm'
        
        switch NetParam.gradientThresholdMethod
            case 'absolute-value' % clip the gradients exceeding the threshold
                NetParam.dL_Wy(NetParam.dL_Wy > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wy(NetParam.dL_Wy < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_by(NetParam.dL_by > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_by(NetParam.dL_by < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wf(NetParam.dL_Wf > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wf(NetParam.dL_Wf < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Uf(NetParam.dL_Uf > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Uf(NetParam.dL_Uf < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Vf(NetParam.dL_Vf > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Vf(NetParam.dL_Vf < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_bf(NetParam.dL_bf > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_bf(NetParam.dL_bf < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wi(NetParam.dL_Wi > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wi(NetParam.dL_Wi < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ui(NetParam.dL_Ui > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ui(NetParam.dL_Ui < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Vi(NetParam.dL_Vi > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Vi(NetParam.dL_Vi < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_bi(NetParam.dL_bi > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_bi(NetParam.dL_bi < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wz(NetParam.dL_Wz > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wz(NetParam.dL_Wz < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Uz(NetParam.dL_Uz > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Uz(NetParam.dL_Uz < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_bz(NetParam.dL_bz > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_bz(NetParam.dL_bz < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wo(NetParam.dL_Wo > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wo(NetParam.dL_Wo < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Uo(NetParam.dL_Uo > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Uo(NetParam.dL_Uo < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Vo(NetParam.dL_Vo > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Vo(NetParam.dL_Vo < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_bo(NetParam.dL_bo > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_bo(NetParam.dL_bo < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ph(NetParam.dL_Ph > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ph(NetParam.dL_Ph < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ch(NetParam.dL_Ch > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ch(NetParam.dL_Ch < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Pc(NetParam.dL_Pc > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Pc(NetParam.dL_Pc < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Cc(NetParam.dL_Cc > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Cc(NetParam.dL_Cc < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Px(NetParam.dL_Px > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Px(NetParam.dL_Px < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Cx(NetParam.dL_Cx > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Cx(NetParam.dL_Cx < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
            case 'l2norm' % clip the gradients whose norm exceed the threshold
                gradientNorm = sqrt(sum(NetParam.dL_Wy(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wy = NetParam.dL_Wy * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_by(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_by = NetParam.dL_by * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wf(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wf = NetParam.dL_Wf * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Uf(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Uf = NetParam.dL_Uf * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Vf(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Vf = NetParam.dL_Vf * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_bf(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_bf = NetParam.dL_bf * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wi(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wi = NetParam.dL_Wi * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ui(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ui = NetParam.dL_Ui * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Vi(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Vi = NetParam.dL_Vi * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_bi(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_bi = NetParam.dL_bi * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wz(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wz = NetParam.dL_Wz * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Uz(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Uz = NetParam.dL_Uz * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_bz(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_bz = NetParam.dL_bz * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wo(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wo = NetParam.dL_Wo * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Uo(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Uo = NetParam.dL_Uo * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Vo(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Vo = NetParam.dL_Vo * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_bo(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_bo = NetParam.dL_bo * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ph(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ph = NetParam.dL_Ph * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ch(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ch = NetParam.dL_Ch * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Pc(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Pc = NetParam.dL_Pc * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Cc(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Cc = NetParam.dL_Cc * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Px(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Px = NetParam.dL_Px * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Cx(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Cx = NetParam.dL_Cx * (NetParam.gradientThreshold / gradientNorm);
                end
        end
        
    case 'gru'
        
        switch NetParam.gradientThresholdMethod
            case 'absolute-value' % clip the gradients exceeding the threshold
                NetParam.dL_Wy(NetParam.dL_Wy > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wy(NetParam.dL_Wy < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_by(NetParam.dL_by > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_by(NetParam.dL_by < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wz(NetParam.dL_Wz > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wz(NetParam.dL_Wz < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Uz(NetParam.dL_Uz > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Uz(NetParam.dL_Uz < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_bz(NetParam.dL_bz > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_bz(NetParam.dL_bz < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wr(NetParam.dL_Wr > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wr(NetParam.dL_Wr < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ur(NetParam.dL_Ur > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ur(NetParam.dL_Ur < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_br(NetParam.dL_br > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_br(NetParam.dL_br < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Wc(NetParam.dL_Wc > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Wc(NetParam.dL_Wc < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Uc(NetParam.dL_Uc > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Uc(NetParam.dL_Uc < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_bc(NetParam.dL_bc > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_bc(NetParam.dL_bc < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ph(NetParam.dL_Ph > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ph(NetParam.dL_Ph < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Ch(NetParam.dL_Ch > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Ch(NetParam.dL_Ch < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Px(NetParam.dL_Px > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Px(NetParam.dL_Px < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
                NetParam.dL_Cx(NetParam.dL_Cx > NetParam.gradientThreshold) = NetParam.gradientThreshold;
                NetParam.dL_Cx(NetParam.dL_Cx < - NetParam.gradientThreshold) = - NetParam.gradientThreshold;
            case 'l2norm' % clip the gradients whose norm exceed the threshold
                gradientNorm = sqrt(sum(NetParam.dL_Wy(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wy = NetParam.dL_Wy * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_by(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_by = NetParam.dL_by * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wz(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wz = NetParam.dL_Wz * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Uz(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Uz = NetParam.dL_Uz * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_bz(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_bz = NetParam.dL_bz * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wr(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wr = NetParam.dL_Wr * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ur(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ur = NetParam.dL_Ur * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_br(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_br = NetParam.dL_br * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Wc(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Wc = NetParam.dL_Wc * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Uc(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Uc = NetParam.dL_Uc * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_bc(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_bc = NetParam.dL_bc * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ph(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ph = NetParam.dL_Ph * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Ch(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Ch = NetParam.dL_Ch * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Px(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Px = NetParam.dL_Px * (NetParam.gradientThreshold / gradientNorm);
                end
                gradientNorm = sqrt(sum(NetParam.dL_Cx(:) .^ 2));
                if gradientNorm > NetParam.gradientThreshold
                    NetParam.dL_Cx = NetParam.dL_Cx * (NetParam.gradientThreshold / gradientNorm);
                end
        end
        
end

end