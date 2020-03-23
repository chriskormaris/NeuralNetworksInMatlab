% DEMO OF MULTI-CLASS CLASSIFICATION USING A NEURAL NETWORK MODEL IN THE MNIST DATASET
clear; close all; 


% Load the MNIST dataset and 
% create the appropriate input and output data matrices 
fprintf('Loading train and test data...');
for j=1:10
   load(['../mnisttxt/train' num2str(j-1) '.txt']);
   load(['../mnisttxt/test' num2str(j-1) '.txt']);
end
fprintf('[OK]\n');


% K: number of classes
K = 10;

T = []; 
X = [];
TtestTrue = []; 
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
figure; 
hold on;
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
    
    % plot some training data
	% the first 10 from each category
    ind = randperm(size(Xtmp,1));
    for i=1:10
        subplot(10,10,10*(j-1)+i);     
        imagesc(reshape(Xtmp(ind(i),:),28,28)');
        axis off;
        colormap('gray');     
    end
%    
end

[N, D] = size(X); % X: (NxD)
