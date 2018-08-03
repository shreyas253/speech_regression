% GMMB_EM_INIT_CMEANS1
%
% initS = gmmb_em_init_cmeans1(data, C)
%
% Create an initialization structure for EM,
% called from gmmb_em, see gmmb_em.
%
% C-means clustering means, uniform weight and covariance
%
% Author(s):
%    Pekka Paalanen <pekka.paalanen@lut.fi>
%
% Copyright:
%
%   Bayesian Classifier with Gaussian Mixture Model Pdf
%   functionality is Copyright (C) 2004 by Pekka Paalanen and
%   Joni-Kristian Kamarainen.
%
%   $Name:  $ $Revision: 1.2 $  $Date: 2004/11/02 09:00:18 $


function initS = gmmb_em_init_cmeans1(data, C);

D = size(data,2);	% dimensions

if C>1
	[lbl, mu] = gmmb_cmeans(data, C, 15);
	% initialization has random nature, results will vary
else
	%lbl = ones(size(data, 1), 1);
	mu = mean(data, 1);
end

% covariances initialization
nsigma = gmmb_covfixer(diag(diag(cov(data))));
sigma = zeros(D,D,C);
for c = 1:C
	sigma(:,:,c) = nsigma;
end

% weights initialization
weight = ones(C,1) * (1/C);

initS = struct(...
	'mu', mu.', ...
	'sigma', sigma, ...
	'weight', weight ...
	);

