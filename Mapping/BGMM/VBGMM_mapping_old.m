function VBGMM_mapping( featDir,featFile,whatDependantModel )
%VBGMM_MAPPING Summary of this function goes here
%   Detailed explanation goes here

close all; clear;

load([featDir '/' featFile]);

%whatDependantModel = 0; % 0-independant; 1-gender dependant; 2-speaker dependant

DATA = cell(noFeats,1);

%% prepare data
mNames = fieldnames(feats.F0.m);
fNames = fieldnames(feats.F0.f);
mNo= length(mNames);
totNo = length(mNames) + length(fNames);

F0=[];
Gain = [];
LSFglot = [];

for i=1:length(mNames)
    F0 = [F0 {feats.F0.m.(mNames{i})'}];
    Gain = [Gain {feats.Gain.m.(mNames{i})'}];
    LSFglot = [LSFglot {feats.LSFglot.m.(mNames{i})'}];   
end
for i=1:length(fNames)
    F0 = [F0 {feats.F0.f.(fNames{i})'}];
    Gain = [Gain {feats.Gain.f.(fNames{i})'}];
    LSFglot = [LSFglot {feats.LSFglot.f.(fNames{i})'}];   
end
F0 = F0';
Gain = Gain';
LSFglot = LSFglot';

clear feats;

%% Modelling
if whatDependantModel==0    
    %% INDEPENDANT MODELS    
    K=100;
    sVals = 1:10;
    for s=1:totNo
        %train data
        sVals_tmp = sVals;
        sVals_tmp(s)=[];
        train_F0 = [F0{sVals_tmp}]; train_F0 = cell2mat(train_F0(:));
        train_Gain = [Gain{sVals_tmp}]; train_Gain = cell2mat(train_Gain(:));
        train_LSFglot = [LSFglot{sVals_tmp}]; train_LSFglot = cell2mat(train_LSFglot(:));
        
        %test data
        test_F0 = [F0{s}]; test_F0 = cell2mat(test_F0(:)); test_F0 = test_F0(:,1:size(test_F0,2)/2);
        test_Gain = [Gain{s}]; test_Gain = cell2mat(test_Gain(:)); test_Gain = test_Gain(:,1:size(test_Gain,2)/2);
        test_LSFglot = [LSFglot{s}]; test_LSFglot = cell2mat(test_LSFglot(:)); test_LSFglot = test_LSFglot(:,1:size(test_LSFglot,2)/2);
        
        % VBGMM
        if s>mNo
            [predVals.F0.f.(fNames{s-mNo}),VBGMM.F0.f.(fNames{s-mNo}),PostPredModel.F0.f.(fNames{s-mNo})] = bayesianGMM( test_F0,train_F0,K );
            [predVals.Gain.f.(fNames{s-mNo}),VBGMM.Gain.f.(fNames{s-mNo}),PostPredModel.Gain.f.(fNames{s-mNo})] = bayesianGMM( test_Gain,train_Gain,K );
            [predVals.LSFglot.f.(fNames{s-mNo}),VBGMM.LSFglot.f.(fNames{s-mNo}),PostPredModel.LSFglot.f.(fNames{s-mNo})] = bayesianGMM( test_LSFglot,train_LSFglot,K );
        else
            [predVals.F0.m.(mNames{s}),VBGMM.F0.m.(mNames{s}),PostPredModel.F0.m.(mNames{s})] = bayesianGMM(test_F0,train_F0,K );
            [predVals.Gain.m.(mNames{s}),VBGMM.Gain.m.(mNames{s}),PostPredModel.Gain.m.(mNames{s})] = bayesianGMM( test_Gain,train_Gain,K );
            [predVals.LSFglot.m.(mNames{s}),VBGMM.LSFglot.m.(mNames{s}),PostPredModel.LSFglot.m.(mNames{s})] = bayesianGMM( test_LSFglot,train_LSFglot,K );        
        end
        save([featDir '/mapped/mapVBGMM_' featFile],'predVals','VBGMM','PostPredModel')
    end    
elseif whatDependantModel==1
    %% GENDER DEPENDANT MODELS    
    K=100;
    sVals = 1:10;
    for s=1:totNo
        %train data
        sVals_tmp = sVals;
        if s>mNo %female
            sVals_tmp([s 1:mNo])=[];
            train_F0 = [F0{sVals_tmp}]; train_F0 = cell2mat(train_F0(:));
            train_Gain = [Gain{sVals_tmp}]; train_Gain = cell2mat(train_Gain(:));
            train_LSFglot = [LSFglot{sVals_tmp}]; train_LSFglot = cell2mat(train_LSFglot(:));
        else %male
            sVals_tmp([s mNo+1:end])=[];
            train_F0 = [F0{sVals_tmp}]; train_F0 = cell2mat(train_F0(:));
            train_Gain = [Gain{sVals_tmp}]; train_Gain = cell2mat(train_Gain(:));
            train_LSFglot = [LSFglot{sVals_tmp}]; train_LSFglot = cell2mat(train_LSFglot(:));
        end
        
        %test data
        test_F0 = [F0{s}]; test_F0 = cell2mat(test_F0(:)); test_F0 = test_F0(:,1:size(test_F0,2)/2);
        test_Gain = [Gain{s}]; test_Gain = cell2mat(test_Gain(:)); test_Gain = test_Gain(:,1:size(test_Gain,2)/2);
        test_LSFglot = [LSFglot{s}]; test_LSFglot = cell2mat(test_LSFglot(:)); test_LSFglot = test_LSFglot(:,1:size(test_LSFglot,2)/2);
        
        % VBGMM
        if s>mNo
            [predVals.F0.f.(fNames{s-mNo}),VBGMM.F0.f.(fNames{s-mNo}),PostPredModel.F0.f.(fNames{s-mNo})] = bayesianGMM( test_F0,train_F0,K );
            [predVals.Gain.f.(fNames{s-mNo}),VBGMM.Gain.f.(fNames{s-mNo}),PostPredModel.Gain.f.(fNames{s-mNo})] = bayesianGMM( test_Gain,train_Gain,K );
            [predVals.LSFglot.f.(fNames{s-mNo}),VBGMM.LSFglot.f.(fNames{s-mNo}),PostPredModel.LSFglot.f.(fNames{s-mNo})] = bayesianGMM( test_LSFglot,train_LSFglot,K );
        else
            [predVals.F0.m.(mNames{s}),VBGMM.F0.m.(mNames{s}),PostPredModel.F0.m.(mNames{s})] = bayesianGMM(test_F0,train_F0,K );
            [predVals.Gain.m.(mNames{s}),VBGMM.Gain.m.(mNames{s}),PostPredModel.Gain.m.(mNames{s})] = bayesianGMM( test_Gain,train_Gain,K );
            [predVals.LSFglot.m.(mNames{s}),VBGMM.LSFglot.m.(mNames{s}),PostPredModel.LSFglot.m.(mNames{s})] = bayesianGMM( test_LSFglot,train_LSFglot,K );        
        end
        save([featDir '/mapped/mapVBGMM_' featFile],'predVals','VBGMM','PostPredModel')
    end 
elseif whatDependantModel==2
    K=50; %maybe??        
    for s=1:totNo
        uttNo = length(F0{s});
        uVals = 1:uttNo;
        for u = 1:uttNo
            %train data
            uVals_tmp = uVals;
            uVals_tmp(uVals_tmp==u)=[];
            train_F0 = [F0{s}(uVals_tmp)]; train_F0 = cell2mat(train_F0);
            train_Gain = [Gain{s}(uVals_tmp)]; train_Gain = cell2mat(train_Gain);
            train_LSFglot = [LSFglot{s}(uVals_tmp)]; train_LSFglot = cell2mat(train_LSFglot);
            
            %test data
            test_F0 = [F0{s}(u)]; test_F0 = cell2mat(test_F0(:)); test_F0 = test_F0(:,1:size(test_F0,2)/2);
            test_Gain = [Gain{s}(u)]; test_Gain = cell2mat(test_Gain(:)); test_Gain = test_Gain(:,1:size(test_Gain,2)/2);
            test_LSFglot = [LSFglot{s}(u)]; test_LSFglot = cell2mat(test_LSFglot(:)); test_LSFglot = test_LSFglot(:,1:size(test_LSFglot,2)/2);
        
            %train VBGMM
            % VBGMM
            if s>mNo
                [predVals.F0.f.(fNames{s-mNo}){u},VBGMM.F0.f.(fNames{s-mNo}){u},PostPredModel.F0.f.(fNames{s-mNo}){u}] = bayesianGMM( test_F0,train_F0,K );
                [predVals.Gain.f.(fNames{s-mNo}){u},VBGMM.Gain.f.(fNames{s-mNo}){u},PostPredModel.Gain.f.(fNames{s-mNo}){u}] = bayesianGMM( test_Gain,train_Gain,K );
                [predVals.LSFglot.f.(fNames{s-mNo}){u},VBGMM.LSFglot.f.(fNames{s-mNo}){u},PostPredModel.LSFglot.f.(fNames{s-mNo}){u}] = bayesianGMM( test_LSFglot,train_LSFglot,K );
            else
                [predVals.F0.m.(mNames{s}){u},VBGMM.F0.m.(mNames{s}){u},PostPredModel.F0.m.(mNames{s}){u}] = bayesianGMM(test_F0,train_F0,K );
                [predVals.Gain.m.(mNames{s}){u},VBGMM.Gain.m.(mNames{s}){u},PostPredModel.Gain.m.(mNames{s}){u}] = bayesianGMM( test_Gain,train_Gain,K );
                [predVals.LSFglot.m.(mNames{s}){u},VBGMM.LSFglot.m.(mNames{s}){u},PostPredModel.LSFglot.m.(mNames{s}){u}] = bayesianGMM( test_LSFglot,train_LSFglot,K );        
            end
            save([featDir '/mapped/mapVBGMM_' featFile],'predVals','VBGMM','PostPredModel')
        end
    end 
end
    

end

