tic
load('env');

datTest = csvread('training.csv', 1, 0); %Read starting second row and the first column
% datTest = csvread('testing.csv');
indexes = datTest(:,1); %Get the index column
datTest(:,1) = []; %Then remove it from the data ???


predictions = getLastRow(datTest); %Save the prediction column.
% manMadeCount = nnz(predictions); %Find the positive classifications
a = size(datTest);
inputCount = a(1); %Store the input count
% notMmCount = inputCount - manMadeCount; %And the negatives
datTest(:,a(2))=[]; %Remove the predictions column.

% datTest = normalise(datTest,ps);

testDimensions=size(datTest);
testCount=testDimensions(1); %Number of test data instances

phiMatrix = calcPhi(datTest,C,testCount);
outputTest=phiMatrix*W.';

same=0;
routputTest = round(outputTest);
for i=1:inputCount
    if predictions(i)==routputTest(i)
        same=same+1;
    end
end
accuracy=same/inputCount*100;

toc



    function m = normalise(dat, ps)
        %Minmax normalises rows.
        %Our data needs to be normalised in coloums.
        m = mapminmax('apply',dat.',ps);
        %Transpose again to get the right orientation.
        m = transpose(m);
    end


    function m = getLastRow(M)
        t = size(M);
        m = M(:,t(2));
    end
