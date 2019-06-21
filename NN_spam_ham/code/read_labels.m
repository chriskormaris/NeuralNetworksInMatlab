% UNUSED

function [ labels ] = read_labels( path )
%READ_LABELS Summary of this function goes here

    labels = [];
    dirData = dir(path);
    for i = 3:length(dirData),
        if length(strfind(dirData(i).name, 'spm')) >= 1,
            labels = [labels 1];
        elseif length(strfind(dirData(i).name, 'msg')) >= 1,
            labels = [labels 0];
        end
    end

end
