function  [y,t_map_main] = mapCall( X,Y,x,x_map_main,ids_map,method,methodOpts,currPath,modPath,modName )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% if settings.concat
%     X = mat2cell(cell2mat(X),lens');
%     Y = mat2cell(cell2mat(Y),lens');
%     x = mat2cell(cell2mat(x),lens');
% end

if strcmp(method,'DNN')
    addpath([currPath '/Mapping/DNN/'])
    DNN_layers = methodOpts.DNN_layers;
    DNN_hidden = methodOpts.DNN_hidden;
    DNN_drpRatio = methodOpts.DNN_drpRatio;    
    dnnOpts = [methodOpts.DNN_layers methodOpts.DNN_hidden methodOpts.DNN_drpRatio];
    x_map = cell(size(x_map_main'));
    for i = 1:length(x_map)
        x_map{i} = x_map_main{i}(ids_map{i},:);
    end
    modName = ['DNNmodel_' modName '.h5'];
    save([modPath '/pyTrain.mat'], 'X', 'Y', 'x', 'x_map', 'dnnOpts', 'modName')
    system(['python ' currPath '/Mapping/DNN/DNN_train.py ' modPath])
    
    load([modPath '/pyPredTest.mat'])
    
    load([modPath '/pyPredMap.mat'])
    t_map_main = x_map_main;
    for i = 1:length(x_map)
        t_map_main{i}(ids_map{i},:) = t{i};
    end

    rmpath([currPath '/Mapping/DNN/'])

elseif strcmp(method,'VBGMM')
    addpath([currPath '/Mapping/VBGMM/'])
    addpath([currPath '/Mapping/VBGMM/VB-GMM/'])
    [y,t_map_main] = bayesianGMM( [X,Y],x,x_map_main,ids_map,methodOpts,1,[modPath 'VBGMMmodel_' modName '.mat'] );    
    rmpath([currPath '/Mapping/VBGMM/'])
    rmpath([currPath '/Mapping/VBGMM/VB-GMM/'])

elseif strcmp(method,'EMGMM')
    addpath([currPath '/Mapping/EMGMM/'])
    addpath([currPath '/Mapping/EMGMM/voicebox/'])
    addpath([currPath '/Mapping/EMGMM/gmmbayestb-v1.0'])
    [y,t_map_main] = crossValEMGMM( X,Y,x,x_map_main,ids_map,methodOpts,1,[modPath 'EMGMMmodel_' modName '.mat'] );
    rmpath([currPath '/Mapping/EMGMM/'])
    rmpath([currPath '/Mapping/EMGMM/voicebox/'])
    rmpath([currPath '/Mapping/EMGMM/gmmbayestb-v1.0']) 
end
end

