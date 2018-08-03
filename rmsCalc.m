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
