% (C) 2018 Shreyas Seshadri
% MIT license
% For license terms and references, see README.txt

function [output_data1, pos_wei1,predicted_feature1] = gmm_prediction_EmGm(input_data1,gmm_model1)


[N1,input_dim1]=size(input_data1);
[model_dim1,M1]=size(gmm_model1.mu); % note assumed to model [inputs outputs] this order

% calculate linear transformations (does not depend on the input data so ->)

A = zeros(model_dim1-input_dim1,input_dim1,M1);

for m = 1 : M1
   
    A(:,:,m) = gmm_model1.Sigma(input_dim1+1:model_dim1,1:input_dim1,m)/gmm_model1.Sigma(1:input_dim1,1:input_dim1,m);
end

% here is the prediction

% for n = 1 : N1
%    
%         input_feature1 = input_data1(n,1:input_dim1);
% 
%         for m = 1 : M1
% 
%             pos_wei1(m) = gmm_model1.weight(m)*gmmb_cmvnpdf(input_feature1, gmm_model1.mu(1:input_dim1,m)', gmm_model1.sigma(1:input_dim1,1:input_dim1,m));
%             predicted_feature1(m,:) = gmm_model1.mu(input_dim1+1:model_dim1,m)+A(:,:,m)*(input_feature1'-gmm_model1.mu(1:input_dim1,m));
%         end
%         
%         output_data1(n,:)=sum(bsxfun(@times,predicted_feature1,pos_wei1'/sum(pos_wei1)),1); 
% end

% this implementation is faster

pos_wei1 = zeros(N1,M1);
predicted_feature1 = zeros(N1,model_dim1-input_dim1,M1);

for m = 1 : M1
    pos_wei1(:,m)=gmm_model1.w(m)*gmmb_cmvnpdf(input_data1,gmm_model1.mu(1:input_dim1,m)',gmm_model1.Sigma(1:input_dim1,1:input_dim1,m))+exp(-700);
    
    predicted_feature1(:,:,m)=bsxfun(@plus,(A(:,:,m)*bsxfun(@minus,input_data1,gmm_model1.mu(1:input_dim1,m)')'),gmm_model1.mu(input_dim1+1:model_dim1,m))';
end

pos_wei1=real(pos_wei1);
%sum(pos_wei1)
pos_wei1=bsxfun(@rdivide,pos_wei1,sum(pos_wei1,2));
assert(isreal(pos_wei1))

output_data1=zeros(N1,model_dim1-input_dim1);


for m = 1 : M1
   output_data1 = output_data1+bsxfun(@times,predicted_feature1(:,:,m),pos_wei1(:,m)); 
end

