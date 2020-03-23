% DEMO OF MULTI-CLASS CLASSIFICATION USING A NEURAL NETWORK MODEL IN THE MNIST DATASET
clc; clear; close all; 

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

rand('state', 0);

% Load the MNIST dataset and 
% create the appropriate input and output data matrices 
fprintf('Loading train and test data...\n');
for j=0:9
   fprintf('Reading "train%g.txt" & "test%g.txt"\n', j, j);
   load(['../../project2017/project2017_1_NN_MNIST/mnisttxt/train' num2str(j) '.txt']);
   load(['../../project2017/project2017_1_NN_MNIST/mnisttxt/test' num2str(j) '.txt']);
   if isOctave,
       fflush(stdout);  % only for Octave
   end
end

fprintf('\n');

% Give the number of hidden units
% test M=100, M=200, M=300, M=400, M=500
prompt = 'Give the number of hidden units, M: ';
M = input(prompt);

activation_function = -1;
% Choose an activation function
while ( size(find([1 2 3]~=activation_function*ones(1,3)), 2) == 3 )
    prompt = 'Choose an activation function,\n1 = log(1 + exp(a)) ,\n2 = (exp(a) + exp(-a)) / (exp(a) - exp(-a)) ,\n3 = cos(a)\n: ';
    activation_function = input(prompt);
end

% Give the regularization parameter lambda
prompt = 'Give the regularization parameter, lambda: ';
lambda = input(prompt);

% Choose batch or mini-batch gradient decsent
batch_or_minibatch = -1;
while ~(batch_or_minibatch == 1 || batch_or_minibatch == 0)
    prompt = 'Input 1 to run batch gradient descent, or 0 to run mini-batch gradient descent: ';
    batch_or_minibatch = input(prompt);
end

if batch_or_minibatch == 1
    % Choose the number of maximum number of iterations of the gradient ascend
    prompt = 'Give the number of max iterations, maxiter: ';
    maxiter = input(prompt);
elseif batch_or_minibatch == 0
    % Choose the number of maximum number of iterations of the gradient ascend
    prompt = 'Give the number of max epochs, maxepochs: ';
    maxepochs = input(prompt);
    % Choose the batch size
    prompt = 'Give the batch size (must be at most equal to the total train data): ';
    batch_size = input(prompt);
end

% Choose size of train and test data
allData = -1;
while ~(allData == 1 || allData == 0)
    prompt = 'Input 1 to use all train and test data, or 0 to use a small subset: ';
    allData = input(prompt);
end

% K: number of classes
K = 10;

T = []; 
X = [];
TtestTrue = []; 
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
for j=1:10
% 
    s = ['train' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);   
    Ntrain(j) = size(Xtmp,1);
    Ttmp = zeros(Ntrain(j), K); 
    Ttmp(:,j) = 1; 
    X = [X; Xtmp]; 
    T = [T; Ttmp]; 
    
    s = ['test' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);
    Ntest(j) = size(Xtmp,1);
    Ttmp = zeros(Ntest(j), K); 
    Ttmp(:,j) = 1; 
    Xtest = [Xtest; Xtmp]; 
    TtestTrue = [TtestTrue; Ttmp]; 
%    
end

% clear train and test sub-matrices to free memory
for j=0:9
   clear(['train' num2str(j)]);
   clear(['test' num2str(j)]);
end

[N, D] = size(X); % X: (NxD)

% gradient descent options
% Max iterations or max epochs
if batch_or_minibatch == 1
    options(1) = maxiter; 
elseif batch_or_minibatch == 0
    options(1) = maxepochs; 
end
% Tolerance 
options(2) = 1e-6; 
% Learning rate 
options(3) = 0.1 / N;
% Batch or mini-batch gradient descent
options(4) = batch_or_minibatch; 

fprintf('\n');

% Normalize the pixels to take values in [0,1].
% Use also mean normalization.
%X = (X - mean(mean(X))) / 255; 
%Xtest = (Xtest - mean(mean(Xtest))) / 255; 
X = X / 255; 
Xtest = Xtest / 255; 

% Add ones vectors as the first column
X = [ones(sum(Ntrain),1), X]; % X: (NxD+1)
Xtest = [ones(sum(Ntest),1), Xtest]; 

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
  options(5) = batch_size;
end

fprintf('\n');

% Initialize W1, W2 for the gradient descent.
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
W1init = [ones(M,1), W1init]; % W1init: (MxD+1)
W2init = [ones(K,1), W2init]; % W2init: (KxM+1)


% DISABLE THIS FOR FASTER EXECUTION!
% Do a gradient check first
%ch = randperm(N); 
%ch = ch(1:20); % select 20 random rows
%disp('Gradcheck for parameters W1, W2.');
%gradcheck_softmaxNN(W1init, W2init, X(ch,:), T(ch,:), lambda, activation_function); 
%fprintf('\n');


if (allData == 0)  % construct a smaller subset of data
    
    % Train the model on a subset of the train data
    % select randomly 6000 rows from X matrix (almost 600 from each category), to train the data 
    ch = zeros(6000, 1);
    for i=1:10
        categoryIndex = (i-1)*6000;
        ch((i-1)*600 + 1 : (i-1)*600 + 600, :) = randperm(6000, 600) + categoryIndex; 
    end
    X = X(ch,:);
    T = T(ch,:);

    % Test the model on a subset of the test data
    % select randomly 1000 rows from Xtest matrix (almost 100 from each category), to test the data
    ch = zeros(1000, 1);
    for i=1:10
      categoryIndex = (i-1)*1000;
      ch((i-1)*100 + 1 : (i-1)*100 + 100, :) = randperm(1000, 100) + categoryIndex; 
    end
    Xtest = Xtest(ch,:);
    TtestTrue = TtestTrue(ch, :);

end

fprintf('The number of the train data is %g.\n', size(X,1));
fprintf('The number of the test data is %g.\n', size(Xtest,1));

fprintf('\n');

% Train the model on all train data
[W1, W2, cost_vector] = ml_softmaxTrain(T, X, lambda, W1init, W2init, options); 

fprintf('\n');

% Test the model on all test data
[Ttest, ~]  = ml_softmaxTest(W1, W2, Xtest); 

fprintf('\n');

[~, Ttrue] = max(TtestTrue,[],2); 
err = length(find(Ttest~=Ttrue)) / size(Xtest,1);
disp(['The error of the method is: ' num2str(err)])

diary off

fig = plot_cost_function(cost_vector);
filename_ext = strcat(filename, '.png');
filepath = strcat('../plots/', filename_ext);
saveas(fig, filepath)
