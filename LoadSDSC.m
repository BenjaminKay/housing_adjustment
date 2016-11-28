function argsout  = LoadSCSC(MatfileName,resultnumber)
%LoadSCSC(MatfileName,resultnumber)
%argsout = LoadSCSC('Task1b.out.mat',3);

%[yPointsVector,PHistory_withadj,GHistory,CleanG,MVec,economy]

load(MatfileName, 'argsout');
%{
yPointsVector = argsout{1};
PHistory_withadj = argsout{2};
GHistory = argsout{3};
CleanG = argsout{4};
MVec = argsout{5};
economy = argsout{6};
%deal?
%}
fprintf('[yPointsVector%i,PHistory_withadj%i,GHistory%i,CleanG%i,MVec%i,economy%i] = argsout{:};\n',[resultnumber,resultnumber,resultnumber,resultnumber,resultnumber,resultnumber]);