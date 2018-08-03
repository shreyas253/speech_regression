function [ xPred,emModel ] = crossValSGMM( X,Y,x,noFolds,K)
%CROSSVALBGMMM Summary of this function goes here
%   Detailed explanation goes here

X_all = [X Y];
[N,~] = size(X_all);

%% Cross Validation

cvError = zeros(length(K),noFolds);
inds = crossvalind('KFold',N,noFolds);

for nF = 1:noFolds
    fprintf('Fold No: %d, Current K = ',nF)
    trainDat = X_all(~(inds==nF),:);    
    testDatIP = X(inds==nF,:);        
    testDatOP = Y(inds==nF,:);  
    for k=1:length(K)    
        fprintf('%d ',k)
        pred = SGMM( trainDat,testDatIP,K(k) );
        cvError(k,nF) = rmsCalc(pred,testDatOP);
    end
    fprintf('\n')
end
cvError = mean(cvError,2);
[~,bestK] = min(cvError);
bestK = K(bestK);

% Final Model for best K
[~,emModel] = SGMM(X_all,[],bestK);
xPred = gmm_prediction_EmGm(x,emModel); 
            
end


function [ x_new ] = MVnorm( x,m,s,key )

if key==1
    x_new = (x-m)/s;
elseif key==0  
    x_new = (x*s)+m;
end

end


function [ rmsPerc ] = rmsCalc( x,y )
%RMSCALC Summary of this function goes here
%   x = vector of predicted
%   y = vector of original
x = x(:);
y = y(:);
sqDiff = (x-y).^2;
mMm = max(y) - min(y);

rmsPerc = sqrt(mean(sqDiff))/mMm*100;
end

