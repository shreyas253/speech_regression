function [ xPred,emClustRes ] = SGMM( X,x,K )
%SGMM - function that returns posterior for a S-GMM
% INPUTS
% x = data with missing features
% X = original concatenated data with full features to cluster
% K = number of clusters (note in Bayesian clustering the number of clusters found can be lower than this)
% OUTPUTS
% xPred     = mean estimate of conditional posterior predictive of the missing features in x, p(xPred|x,X,K)
% emClustRes= SGMM model
% By: Shreyas Seshadri, Ulpu Remes and Okko Rasanen. Last update - 10.10.2016


%% Clustering - VB-DDGMM
if nargin<3
    K=10;
end
%CLUSTERING
[m,v,w] = gaussmix(X,[],[],K,'v');
emClustRes.mu = m';
emClustRes.Sigma = v;
emClustRes.w = w';

%% Conditional Posterior Predictive
if isempty(x)    
    xPred = [];
else
    xPred = gmm_prediction_EmGm(x,emClustRes);
end
end

