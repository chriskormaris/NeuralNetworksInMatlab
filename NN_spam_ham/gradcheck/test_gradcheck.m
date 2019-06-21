clc; close all; diary off;
% add path
addpath('../code')
addpath('../demos')
load_data; % load all the test and train data

% define parameters
M = 2;
activation_function = 1;
lambda = 0.5;

fprintf('M = %g\n', M);
fprintf('activation function = h%g\n', activation_function);
fprintf('lambda = %g\n', lambda);

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

% Do a gradient check
disp('Gradcheck for parameters W1, W2.');
gradcheck_softmaxNN(W1init, W2init, X, T, lambda, activation_function);

rmpath('../code')
rmpath('../demos')
savepath
