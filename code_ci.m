clc
clear all

load data_accent.mat

Y=data(:,1)+1;
X=data(:,2:end);

V=full(ind2vec(Y'))';

N = size(X,1); 
IDX = crossvalind('kfold', N,5); C=[];

A=[] ; 
B=[] ;
for kf=1:1
    %train data
    T1 = X(IDX ~=kf,:)' ; 
    S1 = Y(IDX ~=kf,:)' ; 
    V1 = full(ind2vec(S1)) ; 

    %test data:
    T2 = X(IDX ==kf,:)' ; 
    S2 = Y(IDX ==kf,:)' ; 
    V2 = full(ind2vec(S1)) ; 
    
    net = newff(T1,V1,[8 4]); 
    %net.trainFcn = 'traingd' ;
  
    net.trainParam.epochs = 1000;  % Maximum number of epochs to train.

    net = train(net ,T1,V1) ; 
    V3 = sim (net, T2);
    [temp,S3]= max(V3); 
 
    A = [A; S2'] ; %Real class
    B = [B; S3']; %Estimated class
end
x=A-B;
[m,n]=size(x);
g=0;
for i=1:m
    if x(i)==0
        g=g+1;
    end
end

g=g/m*100
