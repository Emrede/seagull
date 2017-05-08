tic

clc
clear

%Read the training and additional training data
datTraining1 = csvread('training.csv', 1, 0); %Read starting second row and the first column
datTraining2 = csvread('additional_training.csv', 1 ,0);
%Combine the training data in a matrix
datTraining = vertcat(datTraining1, datTraining2);

indexes = datTraining(:,1); %Get the index column
datTraining(:,1) = []; %Then remove index column from the matrix

sizeTrn = size(datTraining);
predictions = datTraining(:,sizeTrn(2)); %Save the prediction column.

%Bilmem ne i$lemi
colMean = nanmean(datTraining); %Get mean of the NaN valued columns
[row,col] = find(isnan(datTraining)); %Get the indexes of NaN valued cells 
datTraining(isnan(datTraining)) = colMean(col); %Change the NaN value with the mean of its column 

[coeff1,score1,latent,tsquared,explained,mu1] = pca(datTraining,'algorithm','eig');
datTraining = score1*coeff1' + repmat(mu1,sizeTrn(1),1);
datTraining(:,sizeTrn(2)) = predictions;
toc

%Convert double matrix into table
VarNames = arrayfun(@(N) sprintf('VarName%d',N), 1:(sizeTrn(2)), 'Uniform', 0);
trnTable = array2table(datTraining, 'VariableNames', VarNames);

%Train a clasifier model with given preferences
md1 = fitcsvm (trnTable, 'VarName4609','Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

[label,score] = predict(md1,trnTable);

bingo = 0;
for i=1:sizeTrn(1)
    if predictions(i)==label(i)
        bingo = bingo+1;
    end
end

%Get an estimate accuracy 
accuracy = bingo/sizeTrn(1)*100;

toc

% % Perform cross-validation
% partitionedModel = crossval(md1.ClassificationSVM, 'KFold', 5);
% 
% % Compute validation accuracy
% validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
% 
% % Compute validation predictions and scores
% [validationPredictions, validationScores] = kfoldPredict(partitionedModel);