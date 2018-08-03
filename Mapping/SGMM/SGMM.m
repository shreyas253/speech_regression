function [ xPred,emClustRes ] = SGMM( X,x,K )
%BAYESIANGMM - function that returns conditional posterior predictive for a VB-GMM
% INPUTS
% x = [nGiven x dimGiven] data with missinkg features
% X = [nOrig x dimOrig] original data with full features to cluster
% K = number of clusters (note in Bayesian clustering the number of clusters found can be lower than this)
% OUTPUTS
% xPred     = mean estimate of conditional posterior predictive of the missing features in x, p(xPred|x,X,K)
% emClustRes= the EM GMM 
% postPred  = the posterior proedictive model p(x|VB-GMM)
% By: Shreyas Seshadri, Ulpu Remes and Okko R?a?nen. Last update - 10.10.2016

% currPath = pwd;

%% Clustering - VB-DDGMM
if nargin<3
    K=10;
end
%  currPath=pwd;
%  addpath([currPath '/gmmbayestb-v1.0/']);
%  addpath([currPath '/EmGm/EmGm/']);
%  addpath([currPath '/voicebox/'])
%CLUSTERING
%emClustRes = gmmb_em(X,'components',K);
%[~,emClustRes1,~] = mixGaussEm(X',K);
[m,v,w] = gaussmix(X,[],[],K,'v');
emClustRes.mu = m';
emClustRes.Sigma = v;
emClustRes.w = w';
% rmpath([currPath '/voicebox/'])
%  rmpath([currPath '/EmGm/EmGm/']);


%% Conditional Posterior Predictive
if isempty(x)    
    xPred = [];
else
    xPred = gmm_prediction_EmGm(x,emClustRes);
    %xPred1 = gmm_prediction_EmGm(x,emClustRes1);
end
%  rmpath([currPath '/gmmbayestb-v1.0/']);
end

