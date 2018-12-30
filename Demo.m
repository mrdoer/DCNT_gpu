function info = Demo()

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', 1)};

run ../../matlab/vl_setupnn ;
% run F:\caffe-master\CREST-Release-master\matconvnetmatlab/vl_setupnn ;
addpath ../../examples ;

opts.expDir = 'exp/' ;
opts.dataDir = 'F:/caffe-master/Data/' ;%F:\caffe-master\Data   exp/data/
opts.modelType = 'tracking' ;
opts.sourceModelPath = 'F:/caffe-master/model/';%'exp/models/' ;
% opts.train = struct() ;
% opts.beta1 = 0.9;
% opts.beta2 = 0.999;
% opts.eps = 1e-8;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = false ;

global resize;
display=1;

g=gpuDevice(1);
clear g;                             

test_seq = 'Deer';
show_plots = true;
[config] = config_list(test_seq);
target_sz = config.gt(1,3:4);
ground_truth = config.gt;

[positions,time] = D_tracking(opts,config,display,varargin);
rects   = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
precisions = precision_plot(positions, ground_truth, test_seq, show_plots);

fps = config.nFrames/time;

fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', test_seq, precisions(20), fps)
       



