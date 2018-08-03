function [ post ] = reformatPostPrior( prior,op )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if strcmp(op.cov_Type,'Full')
    post.v = prior.v0; %1*K
    post.beta = prior.beta0; %1*K
    post.W = prior.W0; % D*D*K
    post.invW = cell(length(post.beta),1);
    for k = 1:length(post.beta)
        post.invW{k} = inv(prior.W0(:,:,k));
    end
    post.m = prior.m0;
end
if strcmp(op.Pi_Type,'DD')
    post.DDalpha = prior.alpha;
end

end

