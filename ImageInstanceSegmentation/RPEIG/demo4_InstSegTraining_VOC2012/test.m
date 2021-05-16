clear; clc; close all;

programTime = clock;
bar = waitbar(0, 'initing...');

addpath('../libs/exportFig');
addpath('../libs/fun4MeanShift');
addpath('../libs/layerExt');
addpath('../libs/myFunctions/');
path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

saveFeaMap = false;
saveFig = false;
% recordInfereneceTime = true;

% set GPU
gpuId = 1;
gpuDevice(gpuId);
%% read matconvnet
meanvalue = reshape([123.6800, 116.7800, 103.9400],[1 1 3]);

path_to_image = 'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\JPEGImages';
modelName = 'pairMMAbsReg_net-epoch-1.mat';

res_dir = './feature_maps';
if ~isdir(res_dir)
    mkdir(res_dir);
end

netbasemodel = load( fullfile('./exp', modelName) );
netbasemodel = netbasemodel.net;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
netbasemodel.move('gpu') ;
netbasemodel.mode = 'normal'; % {test, normal} !! normal --> per-instance normalization
netbasemodel.conserveMemory = 1;

outputLayerName = 'output_l2norm';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(outputLayerName)).outputIndexes).precious = 1;
outputIdx_res7_l2norm = netbasemodel.layers(netbasemodel.getLayerIndex(outputLayerName)).outputIndexes;
%% inference image
val_txt = 'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt';
image_list = importdata(val_txt);
len = length(image_list);
start = 1;
timeList = zeros(len, 1);
waitbar(1/len, bar, 'begin inference...')
for index = start:len
    cur_path_to_image = fullfile(path_to_image, strcat(image_list{index}, '.jpg'));
    cur_path_to_image = strrep(cur_path_to_image,'\','/');
    
    imgOrg = imread(cur_path_to_image);
    sz = size(imgOrg);
    reSZ = round(sz/8)*8;   
    imgOrg = imresize(imgOrg, [reSZ(1), reSZ(2)]);
    im = bsxfun(@minus, single(imgOrg), meanvalue) ;
    inputs = {'data', gpuArray(single(im))};
    startTime = clock;
    netbasemodel.eval(inputs) ;
    feaMap = gather(netbasemodel.vars(outputIdx_res7_l2norm).value);
    endTime = clock;
    useTime = etime(endTime, startTime);
    
    if saveFeaMap
        save(fullfile(res_dir, strcat(image_list{index}, '.mat')), 'feaMap');
    end
%     disp([index, "/", length(image_list), 'done!' ])
    
    if saveFig
        feaMapSize = size(feaMap);
        randRGB4Pred = reshape(feaMap, [], size(feaMap,3));
        rng(777); randProj = randn(feaMapSize(3), 3);
        randRGB4Pred = randRGB4Pred*randProj;
        randRGB4Pred = reshape( randRGB4Pred, [feaMapSize(1), feaMapSize(2), 3] );
        randRGB4Pred = rescaleFeaMap(randRGB4Pred);

        imgFig = figure('visible','off');
        set(imgFig, 'Position', [0 0 800 600]); % [1 1 width height]
        subplot(1,2,1);
        imshow(uint8(imgOrg)); title('original image');
        subplot(1,2,2);
        imagesc(randRGB4Pred); axis off image; title('rand projection onto 3-dim','Interpreter','none');
%     drawnow;
        close(imgFig);
        prefix = './figFolder';
        if ~isdir(prefix)
            mkdir(prefix);
        end
        export_fig(fullfile(prefix, strcat(image_list{index}, '.png')));
    end
    timeList(index) = useTime;
    str = sprintf('computing [%s%%] \n cur inference time %ss, total runing time %ss',  ...
                    num2str(100*index/len), num2str(useTime), num2str(etime(clock, programTime)));
    waitbar((index+1)/len, bar, str);
%     clear randRGB4Pred imgOrg feaMap imgFig
%     if index > 10
%         break
%     end
end
dlmwrite("./useTime.txt", timeList)
close(bar)

