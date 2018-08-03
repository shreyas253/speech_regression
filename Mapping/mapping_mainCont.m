function mapping_mainCont( feats_from,feats_to,dtw_ids,allFeats,config,METADATA,propMain,currPath,featPath,oldFeatPath,silences )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here




% do DTW adjustment and Duration conversion
mappedFeats = cell(size(feats_from));
%v_uv_orig = feats_from(:,allFeats,'');
for n = 1:size(METADATA.data,1)
    if ~isempty(silences)
        silences{n,2} = silences{n,2}(dtw_ids{n});
    end
    for ii = 1:length(allFeats)
        feats_to{n,ii} = feats_to{n,ii}(dtw_ids{n},:); %DTW
        if ~strcmp(allFeats{ii},'v_uv')
            [mappedFeats{n,ii},tmp] = feats_durationConversion(feats_from{n,ii},feats_from{n,ismember(allFeats,'v_uv')},config.voicedRate, config.unvoicedRate, allFeats{ii});
        end
    end
    mappedFeats{n,ismember(allFeats,'v_uv')} = tmp;
end

% get mod feats
if strcmp(config.vocoder,'glott')
    a_tilt = cell(size(METADATA.data,1),1);
    if strcmp(config.slsfOpt,'compensate')
        addpath([currPath '/vocoders/glott/'])
        for n=1:size(METADATA.data,1)
            feats_from{n,ismember(allFeats,'slsf')} = glottLSF_tiltCompensate( feats_from{n,ismember(allFeats,'lsf')}, feats_from{n,ismember(allFeats,'slsf')} );
            feats_to{n,ismember(allFeats,'slsf')} = glottLSF_tiltCompensate( feats_to{n,ismember(allFeats,'lsf')}, feats_to{n,ismember(allFeats,'slsf')} );
            [mappedFeats{n,ismember(allFeats,'slsf')},a_tilt{n}] = glottLSF_tiltCompensate( mappedFeats{n,ismember(allFeats,'lsf')}, mappedFeats{n,ismember(allFeats,'slsf')} );
            
        end
        rmpath([currPath '/vocoders/glott/'])
    end
    ids = getIds(feats_from(:,ismember(allFeats,'v_uv')),feats_from(:,ismember(allFeats,'v_uv')),silences,config.SilOpt);
elseif strcmp(config.vocoder,'straight_sine')    
    mgcAll = mappedFeats(:,ismember(allFeats,'mgc'));
    for n=1:size(METADATA.data,1)
        feats_from{n,ismember(allFeats,'mgc')} = feats_from{n,ismember(allFeats,'mgc')}(:,1:config.mgcOpt);
        feats_to{n,ismember(allFeats,'mgc')} = feats_to{n,ismember(allFeats,'mgc')}(:,1:config.mgcOpt);
        mappedFeats{n,ismember(allFeats,'mgc')} = mappedFeats{n,ismember(allFeats,'mgc')}(:,1:config.mgcOpt);       
    end
    ids = getIds(feats_from(:,ismember(allFeats,'v_uv')),feats_from(:,ismember(allFeats,'v_uv')),silences,'ignore_uv');
end

if config.map.exaggerateData
    [X,Y,ids_all,prIds_all,x_map,mask,featSizes] = getTrainTest(feats_from(:,ismember(allFeats,config.featsToMap)),feats_to(:,ismember(allFeats,config.featsToMap)),ids,mappedFeats(:,ismember(allFeats,config.featsToMap)),mappedFeats(:,ismember(allFeats,'v_uv')),METADATA.data(:,ismember(METADATA.properties,'speakerID')),config.map.ip_Context,config.map.op_Context,config.map.exaggerateData_vals);
else
    [X,Y,ids_all,prIds_all,x_map,mask,featSizes] = getTrainTest(feats_from(:,ismember(allFeats,config.featsToMap)),feats_to(:,ismember(allFeats,config.featsToMap)),ids,mappedFeats(:,ismember(allFeats,config.featsToMap)),mappedFeats(:,ismember(allFeats,'v_uv')),METADATA.data(:,ismember(METADATA.properties,'speakerID')),config.map.ip_Context,config.map.op_Context,[]);    
end

% map
property = METADATA.data(:,ismember(METADATA.properties,propMain));
uniqueProperty = unique(property);
errors = zeros(length(uniqueProperty),1);
for s = 1:length(uniqueProperty)
    bestSp = selectBestSpeakers(uniqueProperty{s},METADATA.data{s,ismember(METADATA.properties,'gender')},config.cont.Cont_training_howMany,config.cont.bestOrd,config.cont.bestOrdGender);
    testIds = ismember(prIds_all,uniqueProperty{s});
    trainIds = ismember(prIds_all,bestSp);
    testIds = ids_all & testIds;
    trainIds = ids_all & trainIds;
    Nold = sum(ids_all & (~testIds));
    X_curr = X(trainIds,:);  Y_curr = Y(trainIds,:);
    x_curr = X(testIds,:);  y_curr = Y(testIds,:);
    x_map_curr = x_map(ismember(property,uniqueProperty{s}));
    mask_curr = mask(ismember(property,uniqueProperty{s}));
    ids_map_curr = mappedFeats(ismember(property,uniqueProperty{s}),ismember(allFeats,'v_uv'));   
    if strcmp(config.normOpt,'spDep')
        prIds_curr = prIds_all(trainIds);
        currProps = unique(prIds_curr);
        mMain = cell(length(currProps),1);
        stdMain = cell(length(currProps),1);
        for ss = 1:length(currProps)
            ids_tmp = ismember(prIds_curr,currProps{ss});
            mMain{ss} = mean(X_curr(ids_tmp,:));
            stdMain{ss} = std(X_curr(ids_tmp,:));
            X_curr(ids_tmp,:) = (X_curr(ids_tmp,:)-repmat(mMain{ss},sum(ids_tmp),1))./repmat(stdMain{ss},sum(ids_tmp),1);
            Y_curr(ids_tmp,:) = (Y_curr(ids_tmp,:)-repmat(mMain{ss},sum(ids_tmp),1))./repmat(stdMain{ss},sum(ids_tmp),1);
        end
        config.methodOpts.mMain = mMain;
        config.methodOpts.stdMain = stdMain;
        config.methodOpts.prIds_curr = prIds_curr;
        config.methodOpts.Nold = Nold;
        
        mTest = mean(x_curr);
        stdTest = std(x_curr);
        x_curr = (x_curr-repmat(mTest,sum(testIds),1))./repmat(stdTest,sum(testIds),1);
        mMain = cell(length(x_map_curr),1); stdMain = cell(length(x_map_curr),1);
        for ss = 1:length(x_map_curr)
            mMain{ss} = mean(x_map_curr{ss}(ids_map_curr{ss},:));
            stdMain{ss} = std(x_map_curr{ss}(ids_map_curr{ss},:));
            x_map_curr{ss}(ids_map_curr{ss},:) = (x_map_curr{ss}(ids_map_curr{ss},:)-repmat(mMain{ss},sum(ids_map_curr{ss}),1))./repmat(stdMain{ss},sum(ids_map_curr{ss}),1);
        end
    elseif strcmp(config.normOpt,'whole')
        %%
    end    
    
    % calling mapping
    [t_curr,t_map_curr] = mapCallCont(X_curr,Y_curr,x_curr,x_map_curr,ids_map_curr,config.method,config.methodOpts,currPath,featPath,oldFeatPath,uniqueProperty{s});
    
    % renorm and calc error and save
    t_curr = (t_curr .* repmat(stdTest,sum(testIds),1)) + repmat(mTest,sum(testIds),1);
    errors(s) = rmsErrorCalc(t_curr,y_curr);
    for ss = 1:size(t_map_curr,1)
        t_map_curr{ss,1}(ids_map_curr{ss},:) = (t_map_curr{ss,1}(ids_map_curr{ss},:) .* repmat(stdMain{s},sum(ids_map_curr{ss}),1)) + repmat(mMain{ss},sum(ids_map_curr{ss}),1);
        tmp = t_map_curr{ss,1}.*mask_curr{ss};
        tmp_uv = t_map_curr{ss,1}(~ids_map_curr{ss},config.map.op_Context*sum(featSizes)+1:(config.map.op_Context+1)*sum(featSizes));
        c = 0;
        tmp_tmp = zeros(size(tmp,1),sum(featSizes),2*config.map.op_Context+1);
        ww = ones(2*config.map.op_Context+1,1);
        for k = -config.map.op_Context:config.map.op_Context
            for j = 1:sum(featSizes)
                %predTest_new(:,j) = predTest_new(:,j)+circshift(predTest_renorm(:,j+c*featdim_out),-k)./(2*wl_output+1);
                tmp_tmp(:,j,k+config.map.op_Context+1) = circshift(tmp(:,j+c*sum(featSizes)),-k).*ww(k+config.map.op_Context+1);
            end
            c = c+1;
        end
        tmp_tmp = nanmean(tmp_tmp,3);
        tmp_tmp(~ids_map_curr{ss},:) = tmp_uv;
        t_map_curr{ss,1} = tmp_tmp;              
        t_map_curr(ss,1:length(featSizes)) = mat2cell(t_map_curr{ss,1},size(t_map_curr{ss,1},1),featSizes');
    end
    mappedFeats(ismember(property,uniqueProperty{s}),ismember(allFeats,config.featsToMap)) = t_map_curr;
    mappedFeats(ismember(property,uniqueProperty{s}),ismember(allFeats,config.featsToMap)) = filteringFunc( mappedFeats(ismember(property,uniqueProperty{s}),ismember(allFeats,config.featsToMap)),config.featsToMapFiltering,config.featsToMapFilteringCoeffts);
    
end
save([featPath '/tmp.mat'],'mappedFeats') 
% save error
checkAndMakeFolders([featPath '/errors/']);
save([featPath '/errors/errors.mat'],'errors');
% reMod feats
if strcmp(config.vocoder,'glott')
    if strcmp(config.slsfOpt,'compensate')
        addpath([currPath '/vocoders/' config.vocoder '/'])
        for n=1:size(METADATA.data,1)
            mappedFeats{n,ismember(allFeats,'slsf')} = glottLSF_tiltRecompensate( mappedFeats{n,ismember(allFeats,'slsf')},  a_tilt{n,1} );
        end
        rmpath([currPath '/vocoders/' config.vocoder '/'])
        for ii = 1:length(allFeats)
            feats = mappedFeats(:,ii);
            checkAndMakeFolders([featPath '/' allFeats{ii} '/']);
            save([featPath '/' allFeats{ii} '/feats.mat'],'feats');
        end
    end
elseif strcmp(config.vocoder,'straight_sine')
    for n=1:size(METADATA.data,1)
        mappedFeats{n,ismember(allFeats,'mgc')} = [mappedFeats{n,ismember(allFeats,'mgc')} mgcAll{n}(:,config.mgcOpt+1:end)];
    end
    for ii = 1:length(allFeats)
        feats = mappedFeats(:,ii);
        checkAndMakeFolders([featPath '/' allFeats{ii} '/']);
        save([featPath '/' allFeats{ii} '/feats.mat'],'feats');
    end
end


end


function ids = getIds(v_uv_form,v_uv_to,silences,opt)
ids = cell(size(v_uv_form));
if strcmp(opt,'ignore_uv')
    for n = 1:length(v_uv_form)
        ids{n} = v_uv_form{n} & v_uv_to{n};        
    end
end
end

function [ rmsPerc ] = rmsErrorCalc( x,y )
%RMSCALC Summary of this function goes here
%   x = vector of predicted
%   y = vector of original
x = x(:);
y = y(:);
sqDiff = (x-y).^2;
mMm = max(y) - min(y);
%mMm = abs(mean(y));
rmsPerc = sqrt(mean(sqDiff))/mMm*100;
end

function x = filteringFunc(x,type,coeffts)
for n = 1:size(x,1)
    for i = 1:size(x,2)
        for ii = 1:size(x{i},2)
            if strcmp(type,'median')
                x{n,i}(:,ii) = medfilt1(x{n,i}(:,ii),coeffts(i));
            end
        end
    end
end
end

function bestSps = selectBestSpeakers(currSp,currGender,noSp,bestSpeakers,SelectGender)
if strcmp(SelectGender,'separate')
    if strcmp(currGender,'m')
        currBest = bestSpeakers{1};
    elseif strcmp(currGender,'f')
        currBest = bestSpeakers{2};
    end
elseif strcmp(SelectGender,'male')
    currBest = bestSpeakers{1};
elseif strcmp(SelectGender,'female')
    currBest = bestSpeakers{2};    
end
currBest(ismember(currBest,currSp)) = [];
bestSps = currBest(1:noSp);
end