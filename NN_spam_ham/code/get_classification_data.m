function [ X, T ] = get_classification_data( files_dir, filenames, labels, no_categories, feature_tokens, testOrTrain )
%GET_CLASSIFICATION_DATA Summary of this function goes here

    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

    N = length(filenames);
    D = length(feature_tokens);
    K = no_categories;
    
    % classification parameter X
    X = zeros(N, D);
    
    
    % reading files
    for i = 1:N,
        if ~isOctave
            fprintf('Reading %s file \"%s\"...\n', testOrTrain, filenames{i});
        else
            fprintf('Reading %s file \"%s\"...\n', testOrTrain, strrep(filenames(i, :), ' ', ''));  % call "strrep" to remove whitespaces
            fflush(stdout);
        end
        
        if ~isOctave
            text = read_file(strcat(files_dir, filenames(i)));        
        else
            text = read_file(strcat(files_dir, strrep(filenames(i, :), ' ', '')));  % call "strrep" to remove whitespaces
        end
        
        if ~isOctave
            text_tokens = strsplit(text);
        else
            text_tokens = strtok(text);
        end
              
        % remove digits, special characters and convert to lowercase
        for k = 1:length(text_tokens),
            text_tokens(k) = lower(text_tokens(k));
            text_tokens(k) = strrep(text_tokens(k), ' ', '');
            text_tokens(k) = regexprep(text_tokens(k), '[0-9]+', '');
        end
        
        % the feature vector contains features with Boolean values
        feature_vector = zeros(1, D);
        for j = 1:D,
            if sum(strcmpi(feature_tokens(j), text_tokens)) >= 1
                feature_vector(j) = 1;
            end
        end
        
        X(i,:) = feature_vector;
    end
    
    fprintf('\n');

    %labels = read_labels(files_dir);
        
    % convert labels to 1-hot matrix T
    T = zeros(N, K);
    for i = 1:length(labels)
        index = logical(labels(i));
        T(i, index+1) = 1;
    end
    
end
