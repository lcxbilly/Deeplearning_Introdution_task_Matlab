
digitDatasetPath = 'MNIST/Train' %fullfile(matlabroot,'toolbox','nnet','nndemos',...
    %'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

testdigitDatasetPath = 'MNIST/Test' %fullfile(matlabroot,'toolbox','nnet','nndemos',...
    %'nndatasets','DigitDataset');
testdigitData = imageDatastore(testdigitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(digitData)

img = readimage(digitData,1);
size(img)

% trainNumFiles = 50000;
[trainDigitData,valDigitData] = splitEachLabel(digitData,0.8,'randomize');
%trainDigitData=digitData;
testDigitData=testdigitData;
%% Define Network Architecture

layers = [
    imageInputLayer(size(img))
    
    convolution2dLayer(3,8,'Padding',1)
    %batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    %batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    %batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


options = trainingOptions('adam',...
    'InitialLearnRate',0.001,...
    'MaxEpochs',40, ...
    'MiniBatchSize', 100, ...
    'Verbose',false,...
    'Plots','training-progress');

%% Train Network Using Training Data
% Train the network using the architecture defined by |layers|, the
% training data, and the training options.  By default, |trainNetwork| uses
% a GPU if one is available (requires Parallel Computing Toolbox(TM) and a
% CUDA-enabled GPU with compute capability 3.0 or higher). Otherwise, it
% uses a CPU. You can also specify the execution environment by using the
% |'ExecutionEnvironment'| name-value pair argument of |trainingOptions|.

%%
% The training progress plot shows the mini-batch loss and accuracy and the
% validation loss and accuracy. For more information on the training
% progress plot, see
% <docid:nnet_examples.mw_507458b6-14c3-4a31-884c-9f2119ff7e05>. The loss
% is the <docid:nnet_ref.bu80p30-3 cross-entropy loss>. The accuracy is the
% percentage of images that the network classifies correctly.
net = trainNetwork(trainDigitData,layers,options);


predictedLabels = classify(net,valDigitData);
valLabels = valDigitData.Labels;

accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

testpredictedLabels = classify(net,testDigitData);
testLabels = testDigitData.Labels;

testaccuracy = sum(testpredictedLabels == testLabels)/numel(testLabels)