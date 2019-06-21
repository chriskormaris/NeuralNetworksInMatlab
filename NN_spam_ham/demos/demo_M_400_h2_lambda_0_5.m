clc; close all; diary off;
% add path
addpath('../code')    

% define parameters
M = 400;
activation_function = 2;
lambda = 0.5;

% Tolerance
options(2) = 1e-6;
% Learning rate
options(3) = 0.5 / N;
% batch or mini-batch gradient ascent
batch_or_minibatch = 0;
options(4) = batch_or_minibatch;
if batch_or_minibatch == 1
	maxiter = 1000;
	options(1) = maxiter;
elseif batch_or_minibatch == 0
	maxepochs = 50;
	options(1) = maxepochs;
	batch_size = 50;
	options(5) = batch_size;
end


% save output data to file
filename = strcat(strcat('M_', int2str(M)), strcat('_h', int2str(activation_function)));
filename = strcat(filename, strcat('_lambda_', num2str(lambda)));
filename_ext = strcat(filename, '.txt');
filepath = strcat('../results/', filename_ext);
if (exist(filepath, 'file'))
  delete(filepath);
end
diary (filepath);
fprintf('M = %g\n', M);
fprintf('activation function = h%g\n', activation_function);
fprintf('lambda = %g\n', lambda);

if batch_or_minibatch == 1
	fprintf('number of max iterations = %g\n', maxiter);
elseif batch_or_minibatch == 0
	fprintf('number of max epochs = %g\n', maxepochs);
end

fprintf('\n');


% Initialize W1, W2 for the gradient ascent.
% Set weights values between [-epsilon_init, epsilon_init].
%epsilon_init = 0.01;
%W1init = randn(M, D) * 2 * epsilon_init - epsilon_init;
%W2init = randn(K, M) * 2 * epsilon_init - epsilon_init;
% ALTERNATIVE Initialize W1, W2 using "sin"
%W1init = reshape(sin(1:M*D), [M, D]) / K;
%W2init = reshape(sin(1:K*M), [K, M]) / K;
% ALTERNATIVE divide by the square root of the 2nd dimension
W1init = randn(M, D) / sqrt(D);
W2init = randn(K, M) / sqrt(M);

% Add ones vectors as the first column
W1init = [ones(M,1), W1init]; % W1init: MxD+1
W2init = [ones(K,1), W2init]; % W2init: KxM+1


% DISABLE THIS FOR FASTER EXECUTION!
% Do a gradient check first
%ch = randperm(N); 
%ch = ch(1:20); % select 20 random rows
%disp('Gradcheck for parameters W1, W2.');
%gradcheck_softmaxNN(W1init, W2init, X(ch,:), T(ch,:), lambda, activation_function); 
%fprintf('\n'); 


fprintf('The number of the train data is %g.\n', size(X,1));
fprintf('The number of the test data is %g.\n', size(Xtest,1));

fprintf('\n');

% Train the model on all train data
[W1, W2, estimate_vector] = ml_softmaxTrain(T, X, lambda, W1init, W2init, options); 

fprintf('\n');

% Test the model on all test data
[Ttest, ~]  = ml_softmaxTest(W1, W2, Xtest); 

fprintf('\n');

Ntest = size(Xtest,1);

[~, Ttrue] = max(TtestTrue,[],2); 
err = length(find(Ttest~=Ttrue)) / Ntest;
disp(['The error of the method is: ' num2str(err)])

diary off

fig = plot_likelihood_estimate(estimate_vector);
filename_ext = strcat(filename, '.png');
filepath = strcat('../plots/', filename_ext);
saveas(fig, filepath)

close all;

rmpath('../code')
savepath
