% DEMO OF MULTI-CLASS CLASSIFICATION USING A NEURAL NETWORK MODEL IN THE MNIST DATASET
clc; clear; close all; diary off;

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

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

% Normalize the pixels to take values in [0,1].
% Use also mean normalization.
%X = (X - mean(mean(X))) / 255; 
%Xtest = (Xtest - mean(mean(Xtest))) / 255; 
X = X / 255; 
Xtest = Xtest / 255;

% Add ones vectors as the first column
X = [ones(sum(Ntrain),1), X]; % X: (NxD+1)
Xtest = [ones(sum(Ntest),1), Xtest]; 
