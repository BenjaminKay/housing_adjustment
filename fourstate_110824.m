%Fit 4 state transition probabilities
clc;
Rhndl       = @(r) r.^(-2:2);
ExpHndl     = @(pvec,r) Rhndl(r)* [pvec(:);1-sum(pvec)];
VarHndl     = @(pvec,r) (Rhndl(r).^2) * [pvec(:);1-sum(pvec)] - ExpHndl(pvec,r)^2;
SkewHndl    = @(pvec,r) ((Rhndl(r) - ExpHndl(pvec,r))/sqrt(VarHndl(pvec,r))).^3 * [pvec(:);1-sum(pvec)];

ExpTrue     = 1.0150383941;
ExpVar      = 0.0087319995;
ExpSkew     = -0.6509681333;

ObjHandle   = @(x)  (ExpTrue - ExpHndl(x(1:4),x(5))) ^ 2  + (ExpVar - VarHndl(x(1:4),x(5)))^2 + (SkewHndl(x(1:4),x(5)) - ExpSkew)^2;


x0 = [.2,.2,.2,.2 , 1.04];
%AX <= b
%b = [1,1,1,1,inf,0,0,0,0,1]';
%A = [[1 0 0 0 0];[0 1 0 0 0];[0 0 1 0 0];[0 0 0 1 0];[0 0 0 0 1];[-1 0 0 0 0];[0 -1 0 0 0];[0 0 -1 0 0];[0 0 0 -1 0];[0 0 0 0 -1]];
%Aeq = [1 1 1 1 0];
%beq = 1;
%options = optimset('Algorithm','interior-point');
options.MaxFunEvals = 2000000;
options.MaxIter = 20000; 
A = [[-1 -1 -1 -1 0];[1 1 1 1 0]];
b = [0;.99];
LB = [0.01;0.01;0.01;0.01;1.01];
UB = [1;1;1;1;5];

%[x fval] = fmincon(ObjHandle,x0,A,b,[],[],[],[],[],options);%
%[x fval] = patternsearch(ObjHandle,x0,[],[],Aeq,beq,[0.01;0.01;0.01;0.01;1.01],[1;1;1;1;5],[],options);
[x fval exitflag] = patternsearch(ObjHandle,x0,A,b,[],[],LB,UB,[],options);
%patternsearch(NegOptHandle,FeasibleGuess,[],[],[],[],LB,UB,nlinconst,options);
[x(1:4),1-sum(x(1:4)),x(5)]
[ExpHndl(x(1:4),x(5)), ExpTrue]
[VarHndl(x(1:4),x(5)), ExpVar]
[SkewHndl(x(1:4),x(5)), ExpSkew]
Rhndl(x(5)) 