% (C) 2018 Shreyas Seshadri
% MIT license
% For license terms and references, see README.txt

function [ xPred,emModel ] = SGMM_adap( X,x,oldModelPath,Nold)
%SGMM_adap - function that adapts previously trained SGMM with new data 
% INPUTS
% x = data with missing features
% X = original concatenated data with full features to cluster
% oldModelPath = path to old model
% Nold = number of datapoints used to train old model
% OUTPUTS
% xPred     = mean estimate of conditional posterior predictive of the missing features in x, p(xPred|x,X,K)
% emClustRes= the EM GMM 
% By: Shreyas Seshadri. Last update - 8.8.2018
[N,D] = size(X);
d = size(x,2); 

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



