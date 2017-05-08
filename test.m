datTest = csvread('testing.csv', 1, 0); %Read starting second row and the first column
indexes = datTest(:,1); %Get the index column
datTest(:,1) = []; %Then remove it from the data ???

sizeDat = size(datTest);
inputCount = sizeDat(1); %Store the input count

%predictions = getLastRow(datTest); %Save the prediction column.
%datTest(:,a(2))=[]; %Remove the predictions column.

%Convert double matrix into table
VarNames = arrayfun(@(N) sprintf('VarName%d',N), 1:(sizeDat(2)), 'Uniform', 0);
testTable = array2table( datTest, 'VariableNames', VarNames);
[label,score] = predict(md1,testTable);

fid = fopen('result.csv', 'w');
fprintf(fid, 'ID,prediction\n');
fclose(fid)

resultTable = horzcat(indexes,label);

%csvwrite('result.csv', concav);
dlmwrite('result.csv',resultTable,'delimiter',',','-append');



