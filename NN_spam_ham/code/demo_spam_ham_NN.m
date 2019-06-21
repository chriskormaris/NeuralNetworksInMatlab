% DEMO OF MULTI-CLASS CLASSIFICATION USING A NEURAL NETWORK MODEL IN A SPAM-HAM DATASET
clc; clear; close all; diary off;

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

rand('state', 0);

% add path
addpath('../code')    

spamTrainDir = '../LingspamDataset/spam-train/';
hamTrainDir = '../LingspamDataset/nonspam-train/';
spamTestDir = '../LingspamDataset/spam-test/';
hamTestDir = '../LingspamDataset/nonspam-test/';
K = 2;  % there are 2 categories

fprintf('Reading feature dictionary...\n');
if isOctave,
    fflush(stdout);  % only for Octave
end

if ~isOctave,
    feature_tokens = strsplit(read_file('../feature_dictionary.txt'));  % for Matlab
elseif isOctave,  
    feature_tokens = strtok(read_file('../feature_dictionary.txt'));  % for Octave
end

fprintf('\nReading TRAIN files...\n');
if isOctave,
    fflush(stdout);  % only for Octave
end

spamTrainFiles = read_filenames(spamTrainDir);
hamTrainFiles = read_filenames(hamTrainDir);
spamTrainLabels = ones(length(spamTrainFiles), 1);
hamTrainLabels = zeros(length(hamTrainFiles), 1);
%Y = [spamTrainLabels; hamTrainLabels];

fprintf('\nReading TEST files...\n');
if isOctave,
    fflush(stdout);  % only for Octave
end

spamTestFiles = read_filenames(spamTestDir);
hamTestFiles = read_filenames(hamTestDir);
spamTestLabels = ones(length(spamTestFiles), 1);
hamTestLabels = zeros(length(hamTestFiles), 1);
%YtestTrue = [spamTestLabels; hamTestLabels];

D = length(feature_tokens);

fprintf('\nConstructing the classification TRAIN and TEST data...\n');
if isOctave,
    fflush(stdout);  % only for Octave
end

[XspamTrain, TspamTrain] = get_classification_data(spamTrainDir, spamTrainFiles, spamTrainLabels, K, feature_tokens, 'train');
[XhamTrain, ThamTrain] = get_classification_data(hamTrainDir, hamTrainFiles, hamTrainLabels, K, feature_tokens, 'train');
[XspamTest, TspamTestTrue] = get_classification_data(spamTestDir, spamTestFiles, spamTestLabels, K, feature_tokens, 'test');
[XhamTest, ThamTestTrue] = get_classification_data(hamTestDir, hamTestFiles, hamTestLabels, K, feature_tokens, 'test');

X = [XspamTrain; XhamTrain];
T = [TspamTrain; ThamTrain];
Xtest = [XspamTest; XhamTest];
TtestTrue = [TspamTestTrue; ThamTestTrue];

N = size(X, 1); % X: (NxD)
Ntest = size(Xtest, 1);

% number of categories: 2, 1 for SPAM and 0 for HAM
K = 2;

% normalize the data using mean normalization
X = X - mean(mean(X));
Xtest = Xtest - mean(mean(Xtest));

% add bias column
X = [ones(N, 1), X];
Xtest = [ones(Ntest, 1), Xtest];
    
fprintf('\nLoading data done!\n');
if isOctave,
    fflush(stdout);  % only for Octave
end

fprintf('\n');

% Give the number of hidden units
% test M=100, M=200, M=300, M=400, M=500
prompt = 'Give the number of hidden units, M: ';
M = input(prompt);

activation_function = -1;
% Choose an activation function
while ( size(find([1 2 3] ~= activation_function * ones(1,3)), 2) == 3 )
    prompt = 'Choose an activation function,\n1 = log(1 + exp(a)) ,\n2 = (exp(a) + exp(-a)) / (exp(a) - exp(-a)) ,\n3 = cos(a)\n: ';
    activation_function = input(prompt);
end

% Give the regularization parameter lambda
prompt = 'Give the regularization parameter, lambda: ';
lambda = input(prompt);

% Choose the number of maximum number of iterations of the gradient ascend
prompt = 'Give number of max iterations, maxiter: ';
maxiter = input(prompt);

% gradient ascent options

% Tolerance 
options(2) = 1e-6; 
% Learning rate 
options(3) = 0.1 / N;

% Tolerance
options(2) = 1e-6;
% Learning rate
options(3) = 0.5 / N;
% batch or mini-batch gradient ascent
batch_or_minibatch = 1;
if batch_or_minibatch == 1
	maxiter = 1000;
	options(1) = maxiter;
elseif batch_or_minibatch == 0
	maxepochs = 50;
	options(1) = maxepochs;
	batch_size = 200;
	options(5) = batch_size;
end

fprintf('\n');

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
W1init = [ones(M,1), W1init]; % W1init: (MxD+1)
W2init = [ones(K,1), W2init]; % W2init: (KxM+1)

% Do a gradient check first
disp('Gradcheck for parameters W1, W2.');
gradcheck_softmaxNN(W1init, W2init, X, T, lambda, activation_function); 

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

filename_ext = strcat(filename, '.png');
filepath = strcat('../plots/', filename_ext);
saveas(plot_likelihood_estimate(estimate_vector), filepath)

rmpath('../code')
savepath
