function phiMatrix = calcPhi(dat,C,trainingCount)
sig = 0.001;
centroidDimensions = size(C);
cRowCount=centroidDimensions(1);
phiRow=zeros(1,cRowCount);
phiMatrix=zeros(trainingCount,cRowCount);
for i=1:trainingCount
    for j=1:cRowCount
        %datRow=dat(i,:); %Debug
        %cRow=C(j,:); %Debug
        
        %Activation function%
        dist = pdist2(dat(i,:),C(j,:),'euclidean'); %distance between X row and indexed centroid
        phi = exp(-1*dist^2/2*sig); %Gausian exp(-1*0.2^2/2)
        phiRow(j)=phi;
    end
    phiMatrix(i,:) = phiRow;
end
end