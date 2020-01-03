function [X_hat, outs] = LRM_CG(D,mask,ins)
% Refer to
%     L. T. Nguyen, J. Kim, S. Kim and B. Shim, "Localization of IoT Networks Via Low-Rank Matrix Completion," 
%     LRM_CG:  Localization in Riemannian manifold using conjugate gradient
% Input:
%     D     :   The original Euclidean distance matrix
%     mask  :  The sampling matrix whoses entries are zeros and ones
%     ins   : The other inputs
%         ins.rank     : the dimension of Euclidean space
%         ins.max_iter : the max iteration
%         ins.tol      : the stopping tolerance
%--------------------------------------------------------------------------
% Output:
%     X_hat : The coordinate matrix of sensor nodes, 
%                each row for each sensor    
%     outs  : The other outputs
%         outs.D_hat      : The reconstructed Euclidean distance matrix
%         outs.cost_D     : 1/n||D_hat - D||_F
%         outs.cost_OmegD : 1/n||P_Omeg(D_hat) - P_Omeg(D)||_F
%--------------------------------------------------------------------------
% Ex:
% 
%     n = 200;    % The number of sensor nodes
%     k = 2;      % The dimension of Euclidean space
%     r = 0.7;    % The radio communication range
% 
%     % Setup the problem
%     noise_setup.addnoise = 0;
%     SNR = nan;           
%     outs_prob_setup = problem_setup(n,k,r,0,noise_setup,SNR);
%     if ~outs_prob_setup.connected_graph
%         display('Can not setup the problem...');
%         display(' ');    
%     end
% 
%     ins.tol = 1e-10;        % The stopping tolerant
%     ins.max_iter = 300;
%     ins.rank = k;
%     [Xhat, outs] = LRM_CG(outs_prob_setup.D,outs_prob_setup.mask,ins); 
% 
%     figure(1)
%     semilogy(1:length(outs.cost_OmegD),outs.cost_OmegD,'LineWidth',2);
%     xlabel('Number of iteration','FontSize',12);
%     ylabel('||P_{\Omega}(D_{hat})-P_{\Omega}(D)||_F','FontSize',12);
%     grid on
% 
%     figure(2)
%     semilogy(1:length(outs.cost_D),outs.cost_D,'LineWidth',2);
%     xlabel('Number of iteration','FontSize',12);
%     ylabel('||D_{hat}-D||_F','FontSize',12);
%     grid on
% 
%--------------------------------------------------------------------------


% Initial setup
%--------------------------------------------------------------------------
d = ins.rank;
tol = ins.tol;
max_iter = ins.max_iter;
D_obs = mask.*D;
[n,~] = size(D_obs);
X1_ = randn(n,d);
X1_ = X1_*X1_';
[s1_,v1_,~] = svd(X1_);
X1.value = X1_;
X1.U = s1_(:,1:d);
X1.Sig = v1_(1:d,1:d);


%--------------------------------------------------------------------------
% Algorithm loop
%--------------------------------------------------------------------------
cost_D = nan(max_iter,1);
cost_OmegD = cost_D;

for i = 1:max_iter  
    
    % Compute the Euclidean gradient
    Egrad = Eu_grad(X1.value,mask,D_obs);

    % Compute the Riemannian gradient
    pxi1 = tangent_orth_proj(X1,Egrad); 
    
    % Compute the conjugate direction 
    if i == 1
        neta1 = Multi_op(pxi1,-1);
    else
        pxi1_ = Trans_op(X0,pxi0,X1);
        neta1_ = Trans_op(X0,neta0,X1);        
        delta1 = pxi1.value - pxi1_.value;        
        h1 = trace(neta1.value'*delta1);        

        % Hager and Zhang 
        beta1 = 1/h1^2*...
           trace(pxi1.value'*(h1*delta1-2*norm(delta1,'fro')^2*neta1.value));
       
        neta1.value = -pxi1.value + beta1*neta1_.value;
        neta1.B = -pxi1.B + beta1*neta1_.B;
        neta1.Qp = -pxi1.Qp + beta1*neta1_.Qp;        

    end    
    
    % Determine an initial step t1 
    inits_t = 10*rand(5,1);
    inits = nan(length(inits_t),1);
    for j = 1:length(inits_t)
        inits(j) = f_X(X1.value + inits_t(j)*neta1.value,mask,D_obs);
    end
    [~,inits_idx] = min(inits);
    t1 = inits_t(inits_idx);

    % Armijo line search method 
    f1 = f_X(X1.value,mask,D_obs);    
    for m = 0:50        
        X1_hat = Retr_op(X1,Multi_op(neta1,(1/2)^m*t1),d);
        f2 = f_X(X1_hat.value,mask,D_obs);
        cond = ...
            (f1 - f2) >= -0.0001*(1/2)^m*t1*trace(pxi1.value'*neta1.value);
        if cond 
            break;          
        end
    end    
    
    X0 = X1;
    X1 = X1_hat; 
    D_hat = 2*(Sym_op(ones(n,1)*diag(X1.value)') - X1.value);      

    if sum(abs(diag(D_hat))) >= 1e-6
        disp('Something''s wrong with the diagonal');
    end 

    cost_D(i) = norm(D_hat - D,'fro')/n;
    cost_OmegD(i) = norm(mask.*(D_hat - D),'fro')/n;    
    
    % Check the stopping tolerance 
    if cost_OmegD(i) <= tol        
        break;
    end
    
    pxi0 = pxi1;
    neta0 = neta1; 
end


if i < max_iter
    cost_D(i+1:max_iter,:) = [];
    cost_OmegD(i+1:max_iter,:) = [];    
end

X_hat = X1.value;
outs.D_hat = D_hat;
outs.cost_D = cost_D;
outs.cost_OmegD = cost_OmegD;
end

%%
function eta0_trans = Trans_op(X0,eta0,X1)
% Input:
%     eta :   a structure
%                 eta.U
%                 eta.Sig
%     X   :   a structure
%                 X.U
%                 X.Sig
%--------------------------------------------------------------------------


A = X1.U'*X0.U;C = X1.U'*eta0.Qp;
eta0_trans.B = A*eta0.B*A' + C*A' + A*C';
eta0_trans.Qp = X0.U*eta0.B*A' + eta0.Qp*A' + X0.U*C' - X1.U*eta0_trans.B;
eta0_trans.value = X1.U*eta0_trans.B*X1.U' +...
    X1.U*eta0_trans.Qp' + eta0_trans.Qp*X1.U';

end

%%
function Y = tangent_orth_proj(X,Z)

A = Sym_op(Z)*X.U;
Y.B = X.U'*A;
Y.Qp = A - X.U*Y.B;
Y.value = X.U*Y.B*X.U' + X.U*Y.Qp' + Y.Qp*X.U';

end

%%
function Y = Sym_op(X)

Y = (X + X')/2;

end

%%
function Y = Retr_op(X,neta,k)
% Input:
%     X      :   a nxn matrix
%     neta   :   a nxn matrix
%     k      :   a number
%--------------------------------------------------------------------------

[Qq,Rq] = qr(neta.Qp);
Qq = Qq(:,1:k);Rq = Rq(1:k,:);
temp = [X.Sig+neta.B Rq';Rq zeros(k)];
[q,ei] = eig(Sym_op(temp));

% supp = 1:length(ei);
for i = 1:length(ei)
    if (abs(ei(i,i)) <= 1e-6) || (abs(imag(ei(i,i))) >= 1e-6)
        ei(i,i) = 0;
%         supp = setdiff(supp,i);
    elseif ei(i,i) < 0
        ei(i,i) = 0;
%         supp = setdiff(supp,i);
    end
end

[~,sort_idx] = sort(diag(ei),'descend');
Y.U = [X.U Qq]*q;
Y.U = Y.U(:,sort_idx(1:k));
Y.Sig = ei(sort_idx(1:k),sort_idx(1:k));
Y.value = Y.U*Y.Sig*Y.U';

%--------------------------------------------------------------------------

end

%%
function Y = Multi_op(X,alpha)

Y.value = alpha*X.value;
Y.B = alpha*X.B;
Y.Qp = alpha*X.Qp;

end


%%
function fX = f_X(X,mask,D_obs)
% Input:
%     X      :   a nxn matrix
%     mask   :   a nxn matrix
%     D_obs  :   a nxn matrix
    
Y = 2*mask.*(Sym_op(ones(length(X),1)*diag(X)') - X) - D_obs;
fX = 0.5*norm(Y,'fro')^2;

end

%%
function Egrad = Eu_grad(X,mask,D_obs)
% Input:
%     X      :   a nxn matrix
%     mask   :   a nxn matrix
%     D_obs  :   a nxn matrix

[n,~] = size(D_obs);
Y = 2*mask.*(Sym_op(ones(n,1)*diag(X)') - X) - D_obs;
Egrad = 2*diag(ones(n,1)'*Sym_op(Y)) - 2*Y;
% Egrad = diag(Y*ones(n,1)) - Y;

end
