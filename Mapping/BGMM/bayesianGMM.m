function  [xPred,model] = bayesianGMM( X,x,methodOpts,modelPath )
%BAYESIANGMM - function that returns conditional posterior predictive for a VB-GMM
% INPUTS
% x = [nGiven x dimGiven] data with missinkg features
% X = [nOrig x dimOrig] original data with full features to cluster
% K = number of clusters (note in Bayesian clustering the number of clusters found can be lower than this)
% OUTPUTS
% xPred     = mean estimate of conditional posterior predictive of the missing features in x, p(xPred|x,X,K)
% vbClustRes= the Bayesian GMM 
% postPred  = the posterior proedictive model p(x|VB-GMM)
% By: Shreyas Seshadri, Ulpu Remes and Okko R?a?nen. Last update - 10.10.2016


[nOrig,dimOrig] = size(X);

%% Clustering - VB-DDGMM
% PRIORS
op.K = methodOpts.VBGMM_K;
prior.alpha = methodOpts.VBGMM_alpha;
op.Pi_Type = methodOpts.VBGMM_Pi_Type;
op.cov_Type = methodOpts.VBGMM_cov_Type;
if strcmp(methodOpts.VBGMM_m0,'all_mean')
    prior.m0 = mean(X,1);
   
end
prior.beta0 = methodOpts.VBGMM_beta0;
if strcmp(methodOpts.VBGMM_W0,'diag_cov')
    prior.W0 = diag(diag(inv(cov(X,1))));     
elseif strcmp(methodOpts.VBGMM_W0,'full_cov')
    prior.W0 = inv(cov(X,1));
end
if strcmp(methodOpts.VBGMM_v0,'dim_plus_2')
    prior.v0 = dimOrig+2; 
end
op.init_Type = methodOpts.VBGMM_init_Type;
op.stopCrit = methodOpts.VBGMM_stopCrit;
op.freethresh = methodOpts.VBGMM_freethresh;
op.max_num_iter = methodOpts.VBGMM_max_num_iter;
op.reorder = methodOpts.VBGMM_reorder;
op.repeats = methodOpts.VBGMM_repeats;


%CLUSTERING
vbClustRes = VB_gmms(X,prior,op); % contains z = found cluster indices
%                                            Nk = sum of posterior responsibilities
%                                            post = posterior model parameters



%% Conditional Posterior Predictive
if strcmp(op.cov_Type,'Full')
%   poterior predictive of VB-GMM with conjugate NIW prior is multivariate-T mixture
    postPred.mu = vbClustRes.post{1}.m;
    postPred.nu = vbClustRes.post{1}.v-dimOrig+1;    
    temp1 = repmat(((vbClustRes.post{1}.beta+1)./(vbClustRes.post{1}.beta .* postPred.nu))',1,dimOrig,dimOrig);
    temp1 = permute(temp1,[3 2 1]);
    temp2 = zeros(size(vbClustRes.post{1}.W));
    for i=1:op.K
        temp2(:,:,i) = inv(vbClustRes.post{1}.W(:,:,i));
    end
    postPred.sigma = temp1 .* temp2;
    postPred.weight = vbClustRes.Nk{1}./sum(vbClustRes.Nk{1});

    dPred = size(x,2);
    [xPred,pos_wei1] = multTmm_prediction(x,postPred); % function very similr to gmm_prediction.m except with mult-T distinution
end

model.postPred = postPred;
model.clustModel = vbClustRes;


end

