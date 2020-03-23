function [Ttest, Ytest]  = ml_softmaxTest(W1, W2, Xtest, activation_function) 
%  
% What it does: It tests an already trained neural network model with regularization
%
% Inputs: 
%         W1: the M x (D+1) dimensional matrix, parameter of matrix W
%         W2: the K x (M+1) dimensional matrix, parameter of matrix W
%         Xtest: Ntest x (D+1) input test data with ones already added in the first column 
% Outputs: 
%         Test:  Ntest x 1 vector of the predicted class labels
%         Ytest: Ntest x K matrix of the sigmoid probabilities     

N = size(Xtest, 1);
M = size(W1, 1);
K = size(W2, 1);

%Z: NxM+1
%Ytest: NxK

% predictions
if nargin == 3 || (nargin == 4 && activation_function == 1)
    Z = h1(Xtest * transpose(W1));
elseif nargin == 4 && activation_function == 2
    Z = h2(Xtest * transpose(W1));
elseif nargin == 4 && activation_function == 3
    Z = h3(Xtest * transpose(W1));
end

Z = [ones(N,1), Z]; % concat ones vector as the first column to Z: (NxM+1)

% softmax probabilities
Ytest = softmax(Z * transpose(W2));

% Hard classification decisions 
[~,Ttest] = max(Ytest,[],2); % (Nx1) vector containing the max of each row

