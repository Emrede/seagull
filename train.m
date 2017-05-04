clc
clear
%%
datTraining1 = csvread('training.csv', 1, 0); %Read starting second row and the first column
datTraining2 = csvread('additional_training.csv', 1 ,0);

datTraining = vertcat(datTraining1, datTraining2);

% A= [0 2 2; 1 NaN 1; 0 NaN 1]; %Debug
colMean = nanmean(datTraining); %Get mean of the NaN valued columns
[row,col] = find(isnan(datTraining)); %Get the indexes of NaN valued cells 
datTraining(isnan(datTraining)) = colMean(col); %Change the NaN value with the mean of its column 

indexes = datTraining(:,1); %Get the index column
datTraining(:,1) = []; %Then remove it from the data ???
predictions = getLastRow(datTraining); %Save the prediction column.
manMadeCount = nnz(predictions); %Find the positive classifications
a = size(datTraining);
inputCount = a(1); %Store the input count
notMmCount = inputCount - manMadeCount; %And the negatives
datTraining(:,a(2))=[]; %Remove the predictions column.

% sigma = 1;
centroidCount = 5;
W=ones(1,centroidCount); %Initial weight values set to 1
learningRate=0.0000003; %0.0000003;

% [datTraining,ps] = normalise(datTraining); %Normalise columns.

[idx, C] = kmeans(datTraining, centroidCount);

trainingDimensions=size(datTraining);
trainingCount=trainingDimensions(1); %Number of training instances

phiMatrix = calcPhi(datTraining,C,trainingCount);

[errorTraining,W,outputTraining] = train(W,phiMatrix,trainingCount,predictions,learningRate);

save('env');


function [m,ps] = normalise(x)
%Minmax normalises rows.
%Data needs to be normalised in coloums.
[x,ps]= mapminmax(x.',0,1);
%Transpose again to get the right orientation.
m=x.';
end

function phiMatrix = calcPhi(dat,C,trainingCount)
phiRow=[];
phiMatrix=[];
centroidDimensions = size(C);
cRowCount=centroidDimensions(1);
for i=1:trainingCount
    for j=1:cRowCount
        %datRow=dat(i,:); %Debug
        %cRow=C(j,:); %Debug
        
        %Activation function%
        dist = pdist2(dat(i,:),C(j,:),'euclidean'); %distance between X row and indexed centroid
        %         phi = exp(-1*dist^2/2*sigma); %Gausian exp(-1*0.2^2/2)
        phiRow(j)=dist;
    end
    phiMatrix = [phiMatrix; phiRow];
end
end

function [errorArray,W,outputArray] = train(W,phiMatrix,sampleCount,target,learningRate)
%global learningRate
errorArray=[];
outputArray=[];
for i=1:sampleCount
    phiRow=phiMatrix(i,:);
    output = phiRow * W.' ; %For debug purposes.
    outputArray = [outputArray,output]; %For debug purposes.
    error = 1/2 *(output - target(i))^2; %Least squares error function
    errorArray = [errorArray , error]; %For debug purposes.
    
    for j=1:length(W)
        W(j)=W(j)-learningRate*error*phiRow(j); %Update weights
    end
end
end

function m = getLastRow(M)
t = size(M);
m = M(:,t(2));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

