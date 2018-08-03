function [Y_new,X_new,idsLog_new,idsCell_new] = exaggerate_trainindata(X,Y,idsLog,idsCell,alpha)
if nargin <3
    alpha = 0.1;
end
   

%mx = mean(cellfun(@mean,x));
%my = mean(cellfun(@mean,y));

%tmp = [];
%for k = 1:length(x)
%   tmp(k,:) = mean((y{k}-x{k})./x{k}); % average difference in features     
%end

%avgdiff = mean(tmp); % average difference

%y_new = cell(size(y));
%for k = 1:length(x)
%    y_new{k} = x{k}+(y{k}-x{k}).*(1+alpha);
%end

%Y_new = X+(Y-X).*(1+alpha);
Y_new = Y;
X_new = X;
idsLog_new = idsLog;
idsCell_new = idsCell;

for k = 1:length(alpha)
    Y_new = [Y_new;X+(Y-X).*(1+alpha(k))];
    X_new = [X_new;X];
    idsLog_new = [idsLog_new;idsLog];
    idsCell_new = [idsCell_new;idsCell];
end
