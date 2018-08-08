% (C) 2018 Shreyas Seshadri
% MIT license
% For license terms and references, see README.txt
function [ rmsPerc ] = rmsCalc( x,y )
%RMSCALC 
%   x = vector of predicted
%   y = vector of original
x = x(:);
y = y(:);
sqDiff = (x-y).^2;
mMm = max(y) - min(y);

rmsPerc = sqrt(mean(sqDiff))/mMm*100;
end
