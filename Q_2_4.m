clear;

dataFolder = fullfile('WARWICK');
imageFolderTrain = fullfile(dataFolder,'train');
labelFolderTrain = fullfile(dataFolder,'trainlabel');
imdsTrain = imageDatastore(imageFolderTrain);

classNames = ["cell" "background"];
labels = [255 0];
pxdsTrain = pixelLabelDatastore(labelFolderTrain,classNames,labels);

pximdsTrain = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
tbl = countEachLabel(pximdsTrain);

numberPixels = sum(tbl.PixelCount);

frequency = tbl.PixelCount / numberPixels;
classWeights = 1 ./ frequency;

inputSize = [128 128 3];
numClasses = numel(classNames);
%%
layers = [
    
    imageInputLayer(inputSize)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    

    transposedConv2dLayer(4,64,'Stride',2,'Cropping',1)
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    
    transposedConv2dLayer(4,64,'Stride',2,'Cropping',1)
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
 
    
    convolution2dLayer(1,numClasses)
    softmaxLayer
    pixelClassificationLayer('Classes',classNames)];


options = trainingOptions('sgdm', ...
    'Momentum',0.95,...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 4, ...
    'InitialLearnRate', 1e-3,...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

net = trainNetwork(pximdsTrain,layers,options);
%%
imageFolderTest = fullfile(dataFolder,'test');
imdsTest = imageDatastore(imageFolderTest);
labelFolderTest = fullfile(dataFolder,'testlabel');
pxdsTest = pixelLabelDatastore(labelFolderTest,classNames,labels);
pxdsPred = semanticseg(imdsTest,net,'WriteLocation',tempdir,'MiniBatchSize',4);
metrics = evaluateSemanticSegmentation(pxdsPred,pxdsTest);

%%  dice
dicecoff=([]);
for i=1:60
similarity = dice(readimage(pxdsPred,i),readimage(pxdsTest,i));
dicecoff(i)=similarity(1);
end
fprintf('dice done\n')
disp(mean(dicecoff))

