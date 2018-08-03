function [X_new,Y_new,ids_all,spIds_all,x_map_new,mask,feat_lens] = getTrainTest(from,to,ids,toMap,idsMap,spIds,w_in,w_out, exxVals)

lens = cellfun(@length,ids);
X = cell2mat(from);
Y = cell2mat(to);
    
ids_all = cell2mat(ids);
spIds_all = [];
for n = 1:length(spIds)
    spIds_all = [spIds_all;repmat(spIds(n),lens(n),1)];
end

if ~isempty(exxVals)
   [Y,X,ids_all,spIds_all] = exaggerate_trainindata(X,Y,ids_all,spIds_all,exxVals); 
end

X_new = [];
for k = -w_in:w_in
   X_new = [X_new circshift(X,k)]; 
end
Y_new = [];
for k = -w_out:w_out
   Y_new = [Y_new circshift(Y,k)]; 
end


lensMap = cellfun(@length,idsMap);
x_map = mat2cell(cell2mat(toMap),lensMap');
x_map_new = cell(size(x_map));
for k = -w_in:w_in
   for j = 1:length(x_map)
      x_map_new{j} = [x_map_new{j} circshift(x_map{j},k)];       
   end
end
mask = cell(size(x_map));
for j = 1:length(x_map)
    masktmp = repmat(idsMap{j},1,size(x_map{j},2));    
    for k = -w_out:w_out     
        mask{j} = [mask{j} circshift(masktmp,k)];
    end
    mask{j}(~idsMap{j},:) = 0;
    mask{j}(~mask{j}) = nan;
end


feat_lens = cellfun(@size,from,'uni',false);
feat_lens = cell2mat(feat_lens(1,:));
feat_lens = feat_lens(2:2:end);