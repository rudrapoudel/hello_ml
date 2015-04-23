function [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures,  patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingWeightCost - given the features in featureMatrix, 
%                         computes the cost and gradient with respect to
%                         the weights, given in weightMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.
    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------
    diff = weightMatrix * featureMatrix - patches;
    temp = diff.^2;
    cost1 = sum(temp(:))/numExamples;
    
    temp = epsilon + groupMatrix * (featureMatrix.^2);
    fmatrix = sqrt(temp);
    cost2 = (lambda/numExamples) * sum(fmatrix(:)); 

%     temp = gamma * abs(weightMatrix);
    cost3 = gamma * sum(weightMatrix(:) .^ 2);
    
%     cost = cost1 + cost3;
    cost = cost1 + cost2 + cost3;

    % gradient term
    grad = 2 * ( ((diff*featureMatrix') ./ numExamples) + (gamma * weightMatrix) );
    grad = grad(:);
    
end