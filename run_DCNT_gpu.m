function results = run_DCNT_gpu(seq,res_path, bSaveImage)

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', [])};

% run ../../matlab/vl_setupnn ;
% addpath ../../examples ;

opts.expDir = 'exp/' ;
opts.dataDir = 'F:\caffe-master\Data\' ;
opts.modelType = 'tracking' ;
opts.sourceModelPath = 'F:\caffe-master\model\';
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

resize = 50;

display = 0;

g=gpuDevice(1);
clear g;                             


config.imgList=seq.s_frames;
config.gt = seq.init_rect;
config.nFrames=seq.len;

img_files = seq.len;

[positions, time] = D_tracking(opts,config,display,varargin); 
% positions = result;

% ================================================================================
% Return results to benchmark, in a workspace variable
% ================================================================================
% rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
% rects(:,3) = positions(3);
% rects(:,4) = positions(4);
rects = positions;
results.type   = 'rect';
results.res    = rects;
results.fps    = numel(img_files)/time;
fprintf('time = %.3f',time);

end

