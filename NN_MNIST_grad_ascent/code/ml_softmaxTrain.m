function [W1, W2, estimate_vector] = ml_softmaxTrain(T, X, lambda, W1init, W2init, options, activation_function)
%
% What it does: It trains using gradient ascent a neural network
%               model with regularization
%
% Inputs: 
%         T: N x K binary output data matrix indicating the classes
%         X: N x (D+1) input data vector with ones already added in the first column
%         lambda: the positive regularizarion parameter
%         Winit: K x (D+1) matrix of the initial values of the parameters 
%         options: options(1) is the maximum number of iterations 
%                  options(2) is the tolerance
%                  options(3) is the learning rate eta 
% Outputs: 
%         W1: the trained M x (D+1) matrix of the parameters     
%         W2: the trained K x (M+1) matrix of the parameters     

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

W1 = W1init;
W2 = W2init;

N = size(X,1);

% Tolerance
tol = options(2);

% Learning rate
eta = options(3);

% batch or minibatch gradient ascent
batch_or_minibatch = options(4);


if batch_or_minibatch == 1
    %% Batch gradient ascent

    disp('Running batch gradient ascent...')
    
    maxiter = options(1);
    Ewold = -Inf;
    estimate_vector = zeros(maxiter, 1);
    for it=1:maxiter

        % Call the cost function to compute both the value of the cost
        % and its gradients. You should store the value of the cost to 
        % the variable Ew and the gradients to the M x (D+1) matrix gradEw1
		    % and the K x (M+1) matrix gradEw2.

        if nargin >= 6
            if nargin == 6
                activation_function = 1;
            end
            [Ew, gradEw1, gradEw2] = costgrad_softmaxNN(W1, W2, X, T, lambda, activation_function);
        end

        % Show the current cost function on screen
        fprintf('Iteration: %d, Cost function: %f\n', it, Ew); 

        % Break if you achieve the desired accuracy in the cost function
        if abs(Ew - Ewold) < tol
            fprintf('|Ew - Ewold|: %g\n', abs(Ew - Ewold)); 
            fprintf('Cost difference less than tolerance has been achieved!\n'); 
            break;
        end

        % Update parameters based on gradient ascent 
        W1 = W1 + eta * gradEw1; 
        W2 = W2 + eta * gradEw2; 

        estimate_vector(it, 1) = Ew;
        Ewold = Ew;
        
        if isOctave,  
            fflush(stdout);
        end
        
    end
    
elseif batch_or_minibatch == 0
    %% Mini-batch gradient ascent

    disp('Running mini-batch gradient ascent...')

    maxepochs = options(1);
    batch_size = options(5);
    s_old = -Inf;
    estimate_vector = zeros(maxepochs, 1);
    for ep=1:maxepochs

        if nargin >= 6
            if nargin == 6
                activation_function = 1;
            end

            % randomly shuffle the dataset (optional)
            random_indices = randperm(size(X,1));
            X = X(random_indices, :);
            T = T(random_indices, :);
            
            s = 0;
            iterations = floor(N / batch_size);
            for i=1:iterations
                start_index = (i-1) * batch_size + 1;
                end_index = i * batch_size;
                xi = X(start_index:end_index, :);
                ti = T(start_index:end_index, :);
                [Ewi, gradEw1i, gradEw2i] = costgrad_softmaxNN(W1, W2, xi, ti, lambda, activation_function);
                s = s + Ewi;

                % Update parameters based on minibatch gradient ascent
                W1 = W1 + eta * gradEw1i;
                W2 = W2 + eta * gradEw2i;
            end
        end

        % Show the current cost function on screen
        fprintf('Epoch: %d, Cost function: %f\n', ep, s); 

        % Break if s has converged
        if abs(s - s_old) < tol
            fprintf('Cost difference less than tolerance has been achieved!\n'); 
            break;
        end

        estimate_vector(ep, 1) = s;
        s_old = s;
        
        if isOctave,  
            fflush(stdout);
        end
        
    end
    
end
