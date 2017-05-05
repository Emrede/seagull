clc
clear

datTraining1 = csvread('training.csv', 1, 0); %Read starting second row and the first column
datTraining2 = csvread('additional_training.csv', 1 ,0);

datTraining = vertcat(datTraining1, datTraining2);
datTraining(:,1) = []; %Remove first column from the data
datSize = size(datTraining);

colMean = nanmean(datTraining); %Get mean of the NaN valued columns
[row,col] = find(isnan(datTraining)); %Get the indexes of NaN valued cells 
datTraining(isnan(datTraining)) = colMean(col); %Change the NaN value with the mean of its column 

VarNames = arrayfun(@(N) sprintf('VarName%d',N), 2:(datSize(2)+1), 'Uniform', 0);
FV_table = array2table( datTraining, 'VariableNames', VarNames);

[trainedClassifier, validationAccuracy] = trainClassifier(FV_table);