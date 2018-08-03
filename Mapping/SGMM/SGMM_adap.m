function [ xPred,emModel ] = SGMM_adap( X,x,oldModelPath,Nold)

[N,D] = size(X);
d = size(x,2); %% this is in general case of mapping(maust be modified to more general case)

load(oldModelPath);
K = length(sgmm_model.w);

emModel = MAPadaptation(sgmm_model,X,Nold,K,D);

xPred = gmm_prediction_EmGm(x,emModel); % function very similr to gmm_prediction.m except with mult-T distinution         

end

function [newModel] = MAPadaptation(oldModel,X,Nold,K,d)
T = size(X,1);

[~,p,~,~] = gaussmixp(X,oldModel.mu',oldModel.Sigma,oldModel.w);
N = sum(p,1);
E_x = X'*p./repmat(N+10^-16,d,1);
E_x2 = zeros(d,d,K);
for k = 1:K
    E_x2(:,:,k) = X'*(repmat(p(:,k),1,d).*X)/(N(k)+10^-16);
end

alpha = N./(N+Nold);

% weight
newModel.w = (alpha.*N/T) + ((1-alpha).*oldModel.w);
newModel.w = newModel.w /sum(newModel.w );
% mean
newModel.mu = (repmat(alpha,d,1).*E_x) + (repmat(1-alpha,d,1).*oldModel.mu);
%sigma
newModel.Sigma = zeros(d,d,K);
for k = 1:K   
    oldMuSq = oldModel.mu(:,k)*oldModel.mu(:,k)';
    newMuSq = newModel.mu(:,k)*newModel.mu(:,k)';
    newModel.Sigma(:,:,k) = (alpha(k)*E_x2(:,:,k)) + ((1-alpha(k))*(oldModel.Sigma(:,:,k)+oldMuSq)) - (newMuSq);    
end
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

