clc; clear; close all; diary off;

format long;

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

rand('state', 0)
% add path
addpath('../code')    

spamTrainDir = '../../LingspamDataset/spam-train/';
hamTrainDir = '../../LingspamDataset/nonspam-train/';
spamTestDir = '../../LingspamDataset/spam-test/';
hamTestDir = '../../LingspamDataset/nonspam-test/';
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
spamTrainLabels = ones(size(spamTrainFiles, 1));
hamTrainLabels = zeros(size(hamTrainFiles, 1));
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

rmpath('../code')
savepath
