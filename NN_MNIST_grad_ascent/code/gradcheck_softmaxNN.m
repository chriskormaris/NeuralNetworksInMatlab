function [diff1, diff2] = gradcheck_softmaxNN(W1, W2, X, T, lambda, activation_function) 
%

D = size(X, 2) - 1;
[M, ~] = size(W1);
[K, ~] = size(W2);

% Compute the analytic gradients and store them in gradEw
%
if nargin >= 5
    if nargin == 5
        activation_function = 1;
    end
    [~, gradEw1, gradEw2] = costgrad_softmaxNN(W1, W2, X, T, lambda, activation_function);
end

% Scan all parameters to compute 
% numerical gradient estimates and store them
% in the matrices numgradEw1, numgradEw2.
epsilon = 1e-6; 
numgradEw1 = zeros(M, D+1);
numgradEw2 = zeros(K, M+1);

% numgradEw for parameter W1
if nargin >= 5
    if nargin == 5
        activation_function = 1;
    end
    for i=1:M
        for j=1:D+1
            W1tmp = W1;
            W1tmp(i,j) = W1(i,j) + epsilon;
            Ewplus = costgrad_softmaxNN(W1tmp, W2, X, T, lambda, activation_function);

            W1tmp = W1;
            W1tmp(i,j) = W1(i,j) - epsilon;
            Ewminus = costgrad_softmaxNN(W1tmp, W2, X, T, lambda, activation_function);
            
            numgradEw1(i,j) = (Ewplus - Ewminus) / (2 * epsilon);
        end
    end
end

% numgradEw for parameter W2
if nargin >= 5
    if nargin == 5
        activation_function = 1;
    end
    for i=1:K
        for j=1:M+1
            W2tmp = W2;
            W2tmp(i,j) = W2(i,j) + epsilon;
            Ewplus = costgrad_softmaxNN(W1, W2tmp, X, T, lambda, activation_function);

            W2tmp = W2;
            W2tmp(i,j) = W2(i,j) - epsilon;
            Ewminus = costgrad_softmaxNN(W1, W2tmp, X, T, lambda, activation_function);
            
            numgradEw2(i,j) = (Ewplus - Ewminus) / (2 * epsilon);
        end
    end
end

% Display the absolute norm as an indication of how close 
% the numerical gradients are to the analytic gradients
diff1 = sum(sum(abs(gradEw1 - numgradEw1))) / sum(sum(abs(gradEw1)));
diff2 = sum(sum(abs(gradEw2 - numgradEw2))) / sum(sum(abs(gradEw2)));

disp(['The maximum absolute norm for parameter W1, in the gradcheck is: ' num2str(diff1) ]);
disp(['The maximum absolute norm for parameter W2, in the gradcheck is: ' num2str(diff2) ]);

