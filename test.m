datTest = csvread('testing.csv', 1, 0); %Read starting second row and the first column
indexes = datTest(:,1); %Get the index column
datTest(:,1) = []; %Then remove it from the data ???

sizeDat = size(datTest);
inputCount = sizeDat(1); %Store the input count

%predictions = getLastRow(datTest); %Save the prediction column.
%datTest(:,a(2))=[]; %Remove the predictions column.

%Convert double matrix into table
VarNames = arrayfun(@(N) sprintf('VarName%d',N), 2:(sizeDat(2)+1), 'Uniform', 0);
FVt_table = array2table( datTest, 'VariableNames', VarNames);
output = trainedClassifier.predictFcn(FVt_table);

fid = fopen('result.csv', 'w');
fprintf(fid, 'ID,prediction\n');
fclose(fid)

concav = horzcat(indexes,output);

%csvwrite('result.csv', concav);
dlmwrite('result.csv',concav,'delimiter',',','-append');

% same=0;
% for i=1:inputCount
%     if predictions(i)==output(i)
%         same=same+1;
%     end
% end
% 
% accuracy=same/inputCount*100;


function m = getLastRow(M)
t = size(M);
m = M(:,t(2));
end