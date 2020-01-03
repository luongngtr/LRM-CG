function outs = problem_setup(n,k,r,disp,noise_setup,SNR)
% Input:
%     n   :   The number of sensor nodes randomly distributed over [0 1]^k
%     k   :   The dimension of the Euclidean space
%     r   :   The radio communication range (0 < r <= 1)
%--------------------------------------------------------------------------   

T = 100;t = 1;
while t < T 
    % The original coordinate matrix
    X = rand(n,k);
%     X = random('unif',1,2,n,k);

    % The Gram matrix
    Y = X*X';

    % The Euclidean distance matrix
    D = ones(n,1)*diag(Y)' + diag(Y)*ones(1,n) - 2*Y;  
    
    % Adding noise    
    D_true = D;
    if noise_setup.addnoise
        % Estimating the variance of D_ij
        kal_ = 1;
        al_ = 10^(-SNR/20)*(((1+kal_)^2+3*(kal_-1)^2)/12)^(-1/2);       
        for i = 1:n-1
            for j = i+1:n
                D(i,j) = D(i,j) + random('unif',-al_*D(i,j),kal_*al_*D(i,j),1,1);
                D(j,i) = D(i,j);
            end
        end       
        outs.SNR = 20*log(norm(D_true,'fro')/norm(D-D_true,'fro'))/log(10);
    end

    % Sensing within radio communication range r
    mask = ones(n);mask(D > r^2) = 0;    

    % Check if the graph is connected
    connected = check_connected_graph(mask);

    if connected
        break;
    else
        if disp ~= 0
            disp('The graph G is not connected, try again...');
        end
    end
    t = t + 1;
end
if t == T
    if disp ~= 0
        disp(['THE GRAPH G IS ALWAYS INCONNECTED, EVEN TRYING ' ...
            num2str(100) ' TIMES!']);
    end
else
    if disp ~= 0
        disp(['THE GRAPH G IS CONNECTED AFTER ' ...
            num2str(t) ' TRIALS!']);
    end
end

% Sampling indices
sampling_idx = find(mask);
[sub_i, sub_j] = ind2sub([n,n],sampling_idx);
sampling_ratio = (nnz(mask)-n)/(n*(n-1));

% Output parameters
outs.X = X;
outs.Y = Y;
outs.D = D;
outs.D_true = D_true;
outs.mask = mask;
outs.sampling_idx = sampling_idx;
outs.sampling_i = sub_i;
outs.sampling_j = sub_j;
outs.sampling_ratio = sampling_ratio;
outs.connected_graph = connected;
end

%%
function connected_graph = check_connected_graph(mask)
n = length(mask);
sets = ones(n,1)*(1:n);
sets = sets.*mask;

connected_graph = 0;
row_idx = 2:n;row_idx_temp = row_idx;
set_1 = sets(1,:);set_1(set_1 == 0) = [];
for j = 1:n
    for i = row_idx_temp
        set_test = sets(i,:);
        set_test(set_test==0) = [];         
        if ~isempty(intersect(set_1,set_test))
            set_1 = union(set_1,set_test);
            row_idx = setdiff(row_idx,i);
        end
    end
    if isempty(row_idx)
        connected_graph = 1;
        break;
    else
        row_idx_temp = row_idx;
    end
end

end

%%













