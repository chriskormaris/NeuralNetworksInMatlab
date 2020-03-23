function [Ew, gradEw1, gradEw2] = costgrad_softmaxNN(W1, W2, X, T, lambda, activation_function)
%

[N, D] = size(X);
D = D - 1;
M = size(W1, 1);
K = size(W2, 1);

% M=100 or M=200 or M=300 or M=400 or M=500

% array sizes
%T (NxK)
%Y (NxK)
%X (NxD+1)
%Z (NxM+1)
%W1 (M,D+1)
%W2 (K,M+1)

if nargin >= 5
	if nargin == 5
		activation_function = 1;
	end
    % choose an activation function among h1, h2 and h3
    if activation_function == 1
        [Z, gradZ] = h1(X * transpose(W1)); % (NxD+1) * transpose(MxD+1) = (NxD+1) * (D+1xM) = NxM
    elseif activation_function == 2
        [Z, gradZ] = h2(X * transpose(W1));
    elseif activation_function == 3
        [Z, gradZ] = h3(X * transpose(W1));
    end
end

Z = [ones(N,1), Z]; % concat ones vector as the first column in Z: (NxM+1)

A = Z * transpose(W2); % (NxM+1) * transpose(KxM+1) = (NxM+1) * (M+1xK) = NxK

% the regularization term
reg = (0.5 * lambda) * (sum(sum(W1(:,2:D+1).^2)) + sum(sum(W2(:,2:M+1).^2)));

% compute the cost function Ew
%Ew = -sum(sum( T .* log(A) )) + reg;

% ALTERNATIVE, it applies the logsumexp trick 
maximum = max(A, [], 2); % it takes the max across rows of the matrix Y
Ew = - ( sum(sum( T .* A )) - sum(maximum) - sum(log(sum(exp(A - repmat(maximum, 1, K)), 2))) ) + reg;

% Compute also the gradients (if nargout > 1) 
%
if nargout > 1
    Y = softmax(A); % Y: NxK

    % S too contains softmax probabilities
    S = Y;
   
    % gradients

    %W1 (MxD+1)
    %W2 (KxM+1)
    %X (NxD+1)
    %Y (NxK)
    %gradZ (NxM)
	
    delta_k = S - T; %NxK
    delta_j = delta_k * W2(:,2:M+1) .* gradZ; % NxK * KxM .* NxM = NxM .* NxM = NxM
    
    gradEw1 = transpose(delta_j) * X + lambda * W1; % transpose(NxM) * (NxD+1) = (MxN) * (NxD+1) = MxD+1
    gradEw2 = transpose(delta_k) * Z + lambda * W2; % transpose(NxK)*(NxM+1)-(KxM+1) = (KxN)*(NxM+1)-(KXM+1) = (KxM+1)-(KxM+1) = KxM+1

end

end

