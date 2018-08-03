function  [y,t_map_main] = mapCallCont( X,Y,x,x_map_main,ids_map,method,methodOpts,currPath,modPath,oldModPath,modName )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% if settings.concat
%     X = mat2cell(cell2mat(X),lens');
%     Y = mat2cell(cell2mat(Y),lens');
%     x = mat2cell(cell2mat(x),lens');
% end

if strcmp(method,'DNN')
    addpath([currPath '/Mapping/DNN/'])

    x_map = cell(size(x_map_main'));
    for i = 1:length(x_map)
        x_map{i} = x_map_main{i}(ids_map{i},:);
    end
    
    oldModPath = [oldModPath '/DNNmodel_' modName '.h5'];
    modName = ['DNNmodelCont_' modName '.h5'];
    save([modPath '/pyTrain.mat'], 'X', 'Y', 'x', 'x_map', 'modName', 'oldModPath')
    system(['python ' currPath '/Mapping/DNN/DNN_train_cont.py ' modPath])
    
    load([modPath '/pyPredTest.mat'])
    
    load([modPath '/pyPredMap.mat'])
    t_map_main = x_map_main;
    for i = 1:length(x_map)
        t_map_main{i}(ids_map{i},:) = t{i};
    end 

    rmpath([currPath '/Mapping/DNN/'])

elseif strcmp(method,'VBGMM')
    addpath([currPath '/Mapping/VBGMM/'])
    addpath([currPath '/Mapping/VBGMM/VB-GMM_cont/'])
    [y,t_map_main] = bayesianGMMCont( [X,Y],x,x_map_main,ids_map,methodOpts,1,[modPath '/VBGMMmodelCont_' modName '.mat'], [oldModPath '/VBGMMmodel_' modName '.mat'] );    
    rmpath([currPath '/Mapping/VBGMM/'])
    rmpath([currPath '/Mapping/VBGMM/VB-GMM_cont/'])

elseif strcmp(method,'EMGMM')
    addpath([currPath '/Mapping/EMGMM/'])
    addpath([currPath '/Mapping/EMGMM/voicebox/'])
    addpath([currPath '/Mapping/EMGMM/gmmbayestb-v1.0'])
    [y,t_map_main] = EMGMMCont( X,Y,x,x_map_main,ids_map,methodOpts,1,[modPath 'EMGMMmodelCont_' modName '.mat'],[oldModPath 'EMGMMmodel_' modName '.mat'] );
    rmpath([currPath '/Mapping/EMGMM/'])
    rmpath([currPath '/Mapping/EMGMM/voicebox/'])
    rmpath([currPath '/Mapping/EMGMM/gmmbayestb-v1.0']) 
end
end

