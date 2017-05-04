function [outputTest] = submit_test()
load('env', 'W', 'C');

datTest = csvread('testing.csv', 1, 0); %Read starting second row and the first column
indexes = datTest(:,1); %Get the index column
datTest(:,1) = []; %Then remove it from the data ???

% manMadeCount = nnz(predictions); %Find the positive classifications
% notMmCount = inputCount - manMadeCount; %And the negatives
% datTest(:,a(2))=[]; %Remove the predictions column.

% datTest = normalise(datTest,ps);

testDimensions=size(datTest);
testCount=testDimensions(1); %Number of test data instances

phiMatrix = calcPhi(datTest,C,testCount);
outputTest = phiMatrix*W.';

routputTest = round(outputTest);

fid = fopen('result.csv', 'w');
fprintf(fid, 'ID,prediction\n');
fclose(fid)

concav = horzcat(indexes,routputTest);

%csvwrite('result.csv', concav);
dlmwrite('result.csv',concav,'delimiter',',','-append');

end

    function m = normalise(dat, ps)
        %Minmax normalises rows.
        %Our data needs to be normalised in coloums.
        m = mapminmax('apply',dat.',ps);
        %Transpose again to get the right orientation.
        m = transpose(m);
    end