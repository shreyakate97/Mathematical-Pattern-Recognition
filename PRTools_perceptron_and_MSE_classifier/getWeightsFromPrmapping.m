function [ W ] = getWeightsFromPrmapping( someMapping )
%getWeights Extract weight vectors from PRTools5 prmapping with stacked or 
%affine mapping_file.

if (strcmp(someMapping.mapping_file,'stacked'))
    
    mappingCell = someMapping.data;
    c = length(mappingCell); % number of classes
    W = zeros(someMapping.size_in+1, c);
    
    for i = 1:c
        mapping_struct = mappingCell{i}.data;
        W(:, i) = [mapping_struct.offset; mapping_struct.rot];
    end
    
elseif strcmp(someMapping.mapping_file,'affine')
    W = [someMapping.data.offset; someMapping.data.rot];
else
    error('Unexpected mapping_file from given prmapping. Mapping has to be affine or staked.')
end

end

