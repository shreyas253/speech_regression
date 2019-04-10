% (C) 2018 Shreyas Seshadri
% MIT license
% For license terms and references, see README.txt

close all; clear;
currPath = pwd;
addpath([currPath '/Mapping/'])

% all data
n = 500;
d_x = 3;
d_y = 1;
X = rand(n,d_x);
Y = rand(n,d_y);

% training data
n_train = 300;
X_train = X(1:n_train,:);
Y_train = Y(1:n_train,:);

% adapataion data
n_adap = 100;
X_adap = X(n_train+1:n_train+n_adap,:);
Y_adap = Y(n_train+1:n_train+n_adap,:);

% test data
n_test = 100;
X_test = X(n_train+n_adap+1:end,:);
Y_test = Y(n_train+n_adap+1:end,:);

%% SGMM
addpath([currPath '/Mapping/SGMM/'])
addpath([currPath '/Mapping/SGMM/voicebox/'])
addpath([currPath '/Mapping/SGMM/gmmbayestb-v1.0/'])

K = [3,5];
noFolds = 2;
% train initial model
[ Y_pred_noadap,sgmm_model ] = crossValSGMM(X_train,Y_train,X_test,noFolds,K);
saveFile = [currPath '/tmpFiles/sgmm_model.mat'];
save(saveFile,'sgmm_model');

% do model adaptation
adap_dat = [X_adap Y_adap];
Y_pred_adap = SGMM_adap( adap_dat,X_test,saveFile,n_train);

% errors
error_sgmm_noadap = rmsCalc(Y_test,Y_pred_noadap)
error_sgmm_adap = rmsCalc(Y_test, Y_pred_adap)


%% BGMM
addpath([currPath '/Mapping/BGMM/'])


% opts for the BGMM
opts.VBGMM_K = 10;
opts.VBGMM_alpha = 1; % alpha for the piror weight distr
opts.VBGMM_Pi_Type = 'DD'; % prior on weight distr
opts.VBGMM_cov_Type = 'Full'; % type of covatriance
opts.VBGMM_m0 = 'all_mean'; % paramater of prior NIW distr
opts.VBGMM_beta0 = 1; % paramater of prior NIW distr
opts.VBGMM_W0 = 'diag_cov'; % paramater of prior NIW distr
opts.VBGMM_v0 = 'dim_plus_2'; % paramater of prior NIW distr
opts.VBGMM_init_Type = 'random'; % inititialisation
opts.VBGMM_stopCrit = 'freeEnergy'; %'number' of runs or 'freeEnergy'
opts.VBGMM_freethresh = 1e-6;
opts.VBGMM_max_num_iter = 600;
opts.VBGMM_reorder = 0;
opts.VBGMM_repeats = 1;

% train initial model
train_dat = [X_train Y_train];
addpath([currPath '/Mapping/BGMM/VB-GMM/'])
[ Y_pred_noadap,bgmm_model ] = bayesianGMM(train_dat,X_test,opts);
rmpath([currPath '/Mapping/BGMM/VB-GMM/'])
saveFile = [currPath '/tmpFiles/bgmm_model.mat'];
save(saveFile,'bgmm_model');

% do model adaptation
adap_dat = [X_adap Y_adap];
addpath([currPath '/Mapping/BGMM/VB-GMM_cont/'])
Y_pred_adap = bayesianGMMCont( adap_dat,X_adap,opts,saveFile);
rmpath([currPath '/Mapping/BGMM/VB-GMM_cont/'])

% errors
error_bgmm_noadap = rmsCalc(Y_test,Y_pred_noadap)
error_bgmm_adap = rmsCalc(Y_test, Y_pred_adap)


%% DNN
addpath([currPath '/Mapping/DNN/'])

% train initial model
dnnOpts = [3,50,0.15];
DNN_X = X_train;
DNN_Y = Y_train;
DNN_x = X_test;
save([currPath '/tmpFiles/pyTrain.mat'], 'DNN_X', 'DNN_Y', 'DNN_x', 'dnnOpts');
DNN_train_file = [currPath '/Mapping/DNN/DNN_train.py'];
tmpPath = [currPath '/tmpFiles/'];
pyRun = ['python ' DNN_train_file ' ' tmpPath];
system(pyRun)
load([tmpPath '/pyPredTest.mat']);
Y_pred_noadap = y;

% do model adaptation
DNN_X = X_adap;
DNN_Y = Y_adap;
DNN_x = X_test;
oldModPath = [currPath '/tmpFiles/DNN_model.h5'];
save([currPath '/tmpFiles/pyTrain.mat'], 'DNN_X', 'DNN_Y', 'DNN_x', 'oldModPath');
DNN_train_file = [currPath '/Mapping/DNN/DNN_train_cont.py'];
tmpPath = [currPath '/tmpFiles/'];
pyRun = ['python ' DNN_train_file ' ' tmpPath];
system(pyRun)
load([tmpPath '/pyPredTest.mat']);
Y_pred_adap = y;

% errors
error_dnn_noadap = rmsCalc(Y_test,Y_pred_noadap)
error_dnn_adap = rmsCalc(Y_test, Y_pred_adap)
