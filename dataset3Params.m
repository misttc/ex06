function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
bestF1 = 0;
bestC = 1;
bestsigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for C = 0.5:0.5:5
	for sigma = 0.1:0.1:1
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		predictions = svmPredict(model, Xval);
		tp = predictions & yval;
    	fp = predictions & (!yval);             % false positive are those not true but flagged by our algorithm
    	fn = (!predictions) & yval;             % false negative are those are true but missed by our algorithm
    	prec = sum(tp) / (sum(tp)+sum(fp));     % calculate precision
    	rec = sum(tp) / (sum(tp)+sum(fn));      % calculate recall
    	F1 = (2*prec*rec) / (prec + rec);

    	if F1 > bestF1
       		bestF1 = F1;
       		bestC = C;
       		bestsigma = sigma;
    	end
    end
end

C = bestC;
sigma = bestsigma;

% =========================================================================

end
