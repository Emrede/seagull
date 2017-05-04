function [outputTest] = test()
load('env');
%%
datTest = csvread('testing.csv', 1, 0); %Read starting second row and the first column
indexes = datTest(:,1); %Get the index column
datTest(:,1) = []; %Then remove it from the data ???

% predictions = getLastRow(datTest); %Save the prediction column.
% manMadeCount = nnz(predictions); %Find the positive classifications
a = size(datTest);
inputCount = a(1); %Store the input count
% notMmCount = inputCount - manMadeCount; %And the negatives
% datTest(:,a(2))=[]; %Remove the predictions column.

% datTest = normalise(datTest,ps);

testDimensions=size(datTest);
testCount=testDimensions(1); %Number of test data instances

phiMatrix = calcPhi(datTest,C,testCount);
outputTest=phiMatrix*W.';

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

    function phiMatrix = calcPhi(dat,C,trainingCount)
        phiRow=[];
        phiMatrix=[];
        centroidDimensions = size(C);
        cRowCount=centroidDimensions(1);
        for i=1:trainingCount
            for j=1:cRowCount
                %datRow=dat(i,:); %Debug
                %cRow=C(j,:); %Debug
                dist = pdist2(dat(i,:),C(j,:),'euclidean'); %distance between X row and indexed centroid
                %         phi = exp(-1*dist^2/2*sigma); %Gausian exp(-1*0.2^2/2)
                phiRow(j)=dist;
            end
            phiMatrix = [phiMatrix; phiRow];
        end
    end

    function m = getLastRow(M)
        t = size(M);
        m = M(:,t(2));
    end
