function VBGMM_mapping( featDir,featFile,whatDependantModel )
%VBGMM_MAPPING Summary of this function goes here
%   Detailed explanation goes here


load([featDir '/' featFile]);

%whatDependantModel = 0; % 0-independant; 1-gender dependant; 2-speaker dependant


featNames = fieldnames(feats);
noFeats = length(featNames);

%% prepare data
mNames = fieldnames(feats.F0.m);
fNames = fieldnames(feats.F0.f);
mNo= length(mNames);
totNo = length(mNames) + length(fNames);

DATA = cell(noFeats,1);
for i=1:length(mNames)
    for ii = 1:noFeats
        DATA{ii} = [DATA{ii} {feats.(featNames{ii}).m.(mNames{i})'}] ;
    end
end

for i=1:length(fNames)
    for ii = 1:noFeats
        DATA{ii} = [DATA{ii} {feats.(featNames{ii}).m.(mNames{i})'}] ;
    end
end
featDims = zeros(noFeats,1);
for ii = 1:noFeats
    DATA{ii} = DATA{ii}';
    featDims(ii) = size(DATA{ii}{1}{1},2);
end
featDims = featDims/6;
clear feats;

%% Modelling
if whatDependantModel==0    
    %% INDEPENDANT MODELS    
    K=100;
    sVals = 1:10;
    orig = cell(noFeats,1);
    pred = cell(noFeats,1);
    for s=1:totNo
        %train data
        sVals_tmp = sVals;
        sVals_tmp(s)=[];
        train = cell(noFeats,1);
        m = cell(noFeats,1);
        sig = cell(noFeats,1);
        for ii = 1:noFeats
            train{ii} = [DATA{ii}{sVals_tmp}]; train{ii} = cell2mat(train{ii}(:)); train{ii}(:,featDims(ii)*4+1:end) = [];
            m{ii} = mean(train{ii},1);
            sig{ii} = std(train{ii},1);        
            train{ii} = (train{ii} - repmat(m{ii},size(train{ii},1),1)) ./ repmat(sig{ii},size(train{ii},1),1); % normalize training data
        end
        
        %test data
        test = cell(noFeats,1);
        for ii = 1:noFeats
            test{ii} = [DATA{ii}{s}]; test{ii} = cell2mat(test{ii}(:)); test{ii}(:,featDims(ii)*4+1:end) = [];
            orig{ii}{s} = test{ii}(:,featDims(ii)*3+1:featDims(ii)*4); % to predict (unnormalized)
            test{ii} = (test{ii} - repmat(m{ii},size(test{ii},1),1)) ./ repmat(sig{ii},size(test{ii},1),1); % normalize test data
            test{ii} = test{ii}(:,1:featDims(ii)*3); % select only input features            
        end
      
        % VBGMM
        if s>mNo
            for ii = 1:noFeats
                [predTmp,VBGMM.(featNames{ii}).f.(fNames{s-mNo}),modTmp] = bayesianGMM( test{ii},train{ii},K );
                modTmp.mFin = m{ii}; modTmp.sigFin = sig{ii};
                PostPredModel.(featNames{ii}).f.(fNames{s-mNo}) = modTmp; 
                predTmp = (predTmp .* repmat(sig{ii}(featDims(ii)*3+1:end),size(predTmp,1),1)) + repmat(m{ii}(featDims(ii)*3+1:end),size(predTmp,1),1);
                predVals.(featNames{ii}).f.(fNames{s-mNo}) = predTmp;
                pred{ii}{s} = predTmp;
            end
        else
            for ii = 1:noFeats
                [predTmp,VBGMM.(featNames{ii}).m.(mNames{s}),modTmp] = bayesianGMM(test{ii}(:,1:featDims(ii)*3),train{ii},K  );
                modTmp.mFin = m{ii}; modTmp.sigFin = sig{ii};
                PostPredModel.(featNames{ii}).m.(mNames{s}) = modTmp;
                predTmp = (predTmp .* repmat(sig{ii}(featDims(ii)*3+1:end),size(predTmp,1),1)) + repmat(m{ii}(featDims(ii)*3+1:end),size(predTmp,1),1);
                predVals.(featNames{ii}).m.(mNames{s}) = predTmp;
                pred{ii}{s} = predTmp;
            end
        end        
    save([featDir '/mapped/mapVBGMM_' featFile],'predVals','VBGMM','PostPredModel')
    end      
    rmsError = rmsErrorCalc(orig,pred);
    save([featDir '/mapped/mapVBGMM_' featFile],'predVals','VBGMM','PostPredModel','rmsError')
elseif whatDependantModel==1
    %% GENDER DEPENDANT MODELS    
    
elseif whatDependantModel==2
    
end

