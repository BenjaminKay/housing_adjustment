function [yPointsVector,Pout,Gout,CleanG,MVec,economy] = flavinopt_110829b(InitialYPoints,Pt,yMin,yMax,BigT)
%[yPointsVector,PHistory_withadj,GHistory,CleanG,MVec,economy] = flavinopt_110818(InitialYPoints,Pt,yMin,yMax,BigT)
%{
%Dependencies to run on sdsc
'run_opt_on_sdsc_110818d.m', 
	'getSchedule.m',
    'HouseStateProcessor_110826.m'
    'CheckPolicy_110822.m'
    'flavinopt_110822b.m',
        'OptimizeValueSct_110822.m',
            'beninterp2_for_mat.mexa64',
            'meshvecs.m',
        'howard_until_converge_110824.m',
            'beninterp1_for_mat.mexa64'            
        'OptimizeValueSct_continuous_110822.m',        
            'beninterp2_for_mat.mexa64',
%}
% [yPointsVector,PHistory_withadj,GHistory,CleanG,MVec,economy] = flavinopt_110817(InitialYPoints,Pt,yMin,yMax,BigT)
%% Progress Calendar
%7/28/11
    %I've implemented a continuous optimization function that uses a global
    %method. I'm getting some improvement and it takes about three minutes a
    %step (in T). I'm going to integrate it, then cut down on my initialization
    %points and number of reoptimizations and then see if I can get a faster
    %convergence. Also, the SDSC seems to have the global optimization toolkit,
    %so that's good. 
    
%8/03/11
    %Continuous step using pattern search is pretty fast and working. 
    %I'm moved to just allowing 4o points and removed the not start of scratch
    %option so that I can make everything easier. Instead I'm passing Pt, yMin,
    %and yMax from the command line. 500 points takes just 25 minutes 
    
    %Want to check the end points of the s-S region to make sure that my choice
    %of y values is broad enough that I calculate a policy for values outside of
    %the interval. This should just consist of calculating the up and down
    %y results in y_{t+1} of the best and worst states in the s-S region
    %and checking the maximum range (yMin and yMax).  

%8/05/11
    %I think I understand the month problem. When you shift to months, you can
    %have much more leverage because the worst down positions of the market
    %isn't as bad. This allows you to have more housing. 
    
%8/8/11 
    %I still seem to be having a convergence problem with monthly data. However,
    % I did seem to sucessfully implement the matrix interpolation so that I can 
    % do the tauchen method for an arbitary # of points. 
    
    %I may want to store the number of states in the economy object to save
    %time. Since it currently is roughtly as fast in the discrete case and
    %faster in the continuous case, I may not bother.  I confirmed it matched
    %the old results (a bit faster too) and works for 5 discrete points as well.
    
    %Can I do a price change system for housing as well?
    
%8/9/11
    %It was brutal but I've gotten the discrete optimiation to work (or at least
    %it looks like it does. It is slower than before obviously, but not bad at
    %all.
    
%8/16/11
    %Howard part implemented 
    %Reimplemeted a matrix version for faster calcaulation and as setup for
    %continuous part. 
    %To do
    %Implement continuous part. 
%8/18/11    
    %Continious part working'

%8/18/11    
    %Small bug in howard step fixed
    %Realized I used nominal and not real returns which I fixed

%8/21/11    
    %Standardized the way the the valid points are checked and made a function to check it in several places. It appears to give nice results now.      

%8/23/11    
    %Fixed some pricing stuff (now 113 is recomended)   
    %Fixed and standardized the way that the housing state variables are created
    %Fixed the stock calibration to match the Famma French data. 
%8/24/11    
    %Would like to move to a 5 state transition instead of 3.
    %Trans 0.213454437255859   0.048060607910156   0.175162506103516 0.509463500976562  puu implied
    %Return 1.077292480468750
%8/25 - 8/27 various improvements and fixes, including both more housing and stock states.

%8/28/11 higher borrowing rate to decrease leverage. Also want to try
%higher transaction cost. Using Freadie Mac Data, trying 4.2% borrowing
%rate



%% Setup Economy


tic;
    fprintf('Set model parameters\n');
   
    %Model parameters for financial markets
    %Real data P(up) is 72% and RtUp = 1.166 RtDown = .861
    PlcyResPt = 50;
    MinPlcyResPt  = 25;
  
    
    %Yearly 4 pt
    %Real
    %Stock market returns
    %RVec = [0.764478140986761,0.980380610612589,1.19628308023842,1.41218554986425];
    %PVec = [0.122753184410497,0.377246815589503,0.377246815589503,0.122753184410497];
    RVec = [0.702036072638499,0.856554381753301,1.0110726908681,1.1655909999829,1.32010930909771,1.47462761821251];    
    PVec = [0.04822638819224,0.154718157800014,0.297055454007746,0.297055454007747,0.154718157800013,0.04822638819224];

    HouseStates     = 15; %>=3, %test with 4, 9 in production with new calibration
    Rf              = 1.042; % Freadie Mac 1992 - 2011 Conforming. 
   %Rf              = 1.0097302061; %Fama french Rf 1950-2010    
    StockStates = numel(RVec);    
    %Housing PRices

    RhUp                            = 1.077292480468750; %CS 1987-2010 
    %Note that this is calibrated using fourstate_110824.m to match the mean,
    %var and skew of Case Shiller data for 10 city index. 
    Pdd                             = 0.213454437255859;
    Pd                              = 0.048060607910156;
    Pflat                           = 0.175162506103516;    
    Pu                              = 0.509463500976562;    
    Puu                             = 1 - Pu - Pflat - Pd - Pdd;

    PeakToTrough                    = 1 - 0.3295233892;
    MinTransitionstoWorst           = ceil(log(PeakToTrough) / log(RhUp ^ -2));
    MinStatesAboveStartingHouse     = 2;
    MinStatesBelowEndingHouse       = 2;

    
    if HouseStates < MinStatesAboveStartingHouse + MinStatesBelowEndingHouse + MinTransitionstoWorst
       NewHouseStates =  (MinStatesAboveStartingHouse + MinStatesBelowEndingHouse + MinTransitionstoWorst);
       fprintf('Override number of house states from %i to %i in order to have enough room on both sides \n', [HouseStates,NewHouseStates])
       HouseStates = NewHouseStates;
    else
        fprintf('Using %if house price states \n.', HouseStates)
    end    
    
    HPriceVec                       =  (RhUp .^ (-(HouseStates - MinStatesAboveStartingHouse-1):MinStatesAboveStartingHouse)) * Pt;
    StartingHouseIndex              = HouseStates - MinStatesAboveStartingHouse ;
    %HPriceVec(StartingHouseIndex)
    

    
    %Make the state transition matrix
    HTransProbVec           =  [Pdd,Pd,Pflat,Pu,Puu];   
    HTransProbStates        = numel(HTransProbVec);
    
    if HTransProbStates == 3
        HTransProb              = diag(repmat(HTransProbVec(1),numel(HPriceVec)-1,1),-1)+diag(repmat(HTransProbVec(2),numel(HPriceVec),1),0)+diag(repmat(HTransProbVec(3),numel(HPriceVec)-1,1),1);
        HTransProb(1,1)     = HTransProb(1,1) + HTransProbVec(1);
        HTransProb(end,end) = HTransProb(end,end) + HTransProbVec(3);
        
    elseif HTransProbStates == 5
        HTransProb          = diag(repmat(HTransProbVec(1), numel(HPriceVec)-2,1), -2) + ...
                              diag(repmat(HTransProbVec(2), numel(HPriceVec)-1,1), -1) + ...
                              diag(repmat(HTransProbVec(3), numel(HPriceVec)  ,1), 0 ) + ...
                              diag(repmat(HTransProbVec(4), numel(HPriceVec)-1,1), 1 ) + ...
                              diag(repmat(HTransProbVec(5), numel(HPriceVec)-2,1), 2 );
                          
        HTransProb(1,1)         = HTransProb(1,1) + HTransProbVec(1) + HTransProbVec(2);
        HTransProb(2,1)         = HTransProb(2,1) + HTransProbVec(1);
        HTransProb(end-1,end)   = HTransProb(end-1,end) + HTransProbVec(end);        
        HTransProb(end,end)     = HTransProb(end,end) + HTransProbVec(end-1) + HTransProbVec(end);
            
    else
        fprintf('Danger\n')
    end

    %Average price mean((economy.HTransProb ^ 500) * economy.HPriceVec') = 135    
    

    
    HUpDownProb     =  [[0;0;diag(HTransProb,-2)],[0;diag(HTransProb,-1)],diag(HTransProb,0),[diag(HTransProb,1);0],[diag(HTransProb,2);0;0]];


    
    beta            = 0.98; %.98 in FN paper
    HPriceVec       = reshape(HPriceVec, 1, 1, []);
   

    
    
    %Fill in old two state values for economy struct for comparison with other
    %outcomes.
    RtUp    = max(RVec); %sum((RVec >= 1) .* RVec.*PVec) / sum((RVec >= 1) .* PVec);
    RtDown  = min(RVec); %sum((RVec < 1) .* RVec.*PVec) / sum((RVec < 1) .* PVec);
    PeUp = sum((RVec >= 1) .* PVec);
    RtExpected = sum(RVec .* PVec);
    RtVar = sum(PVec .* (RVec - RtExpected).^2);    
    RtGeo = prod(RVec.^PVec);
    
    %Model parameters for preferences with Values from table 1 of F&N 08
    %>1 in paper;  gamma>=0 
        gamma   = 1.015; %1.015 in paper 
    %We want 1-rho to be alpha in the normal model.  Rho >0, 
        rho 	= 1.8;          a = 1-rho; %bboard small a

    %Alpha controls subsitution between two goods. Alpha is <=1.
        alpha   = -6.7; %or perhaps        Alpha = -8;

        lambda  = .05;  %lambda = .05; 

    %Relative price of a unit of housing services
    %want this much greater than 1. For now set to one because this
    %will change the right region of yt to examine and I don't want
    %to have to error check this at the same time as fixing the
    %timing of the housing decision in the felicity function. 
    %Run for this many periods                            

    FinalPolicyTol      = 1e-8;
    BigTol              = .01;
    bn_tune             = [.9]; %on 0 to inf, smaller is more points arround estimated value. Recommend less than 1.     
    RefineGridMax       = 6;   %How many zero policy changes before refining grid?    
    extrapval = -10000;
    %Start Figure numbering at the following value
    %Initialize the difference norm.
    NumberYPoints       = InitialYPoints ;
    timeelabsed         = 0;
    %A structure to pass economic variables
    economy             = struct('RtUp',RtUp,'RtDown',RtDown,'PeUp',PeUp,'Rf',Rf,'Pt',Pt, ...
                                'gamma',gamma,'rho',rho,'a',a,'alpha',alpha, ...
                                'beta',beta,'lambda',lambda,'yMin',yMin,'yMax',yMax,'FinalPolicyTol',FinalPolicyTol, ...
                                'BigTol',BigTol,'bn_tune',bn_tune,'extrapval',extrapval,'CurrPlcyResPt',PlcyResPt, ...
                                'InitialYPoints',InitialYPoints,'extrapvalmin',extrapval, 'RVec',RVec,'PVec',PVec, ...
                                'RtExpected', RtExpected,'RtVar',RtVar,'RtGeo',RtGeo,'MaxPlcyResPt',PlcyResPt, ...
                                'StockStates',StockStates,'HTransProb',HTransProb, 'HPriceVec', HPriceVec, ...
                                'HouseStates',HouseStates,  ...
                                'HUpDownProb', HUpDownProb, 'StartingHouseIndex', StartingHouseIndex, ...
                                'HTransProbStates', HTransProbStates, 'MinStatesAboveStartingHouse', MinStatesAboveStartingHouse,...
                                'MinStatesBelowEndingHouse', MinStatesBelowEndingHouse, 'TransitionstoWorst', MinTransitionstoWorst  ...
                                );

toc 

%% Set up the handles and data structures for optimization
tic;

%This is the value when the value function takes too low a value as inputs
%or negative consumption or housing values. 

    
%Intensive form of utility function
    IntensiveUtilForHandle = @(ct,htplus1,rho,alpha,gamma) ((1/(1-rho)).*(gamma*htplus1.^alpha+ct.^alpha).^((1-rho)/alpha));
 
%This handle ensure that the utility function has positive inputs and if
%not it puts in a penalty value                     
    IntensiveUtilHandle = @(ct,htplus1) IntensiveUtilForHandle(ct+eps,htplus1+eps,rho,alpha,gamma) .* (ct>0 & htplus1>0) ...
                                    + not(ct>0 & htplus1>0)*extrapval*10;      
     
%Set up y points. No reason not to evenly space these. yt = Wt / Ht - lambda * Pt


    %LogyMin = log(yMin); LogyMax = log(yMax);
    
    %linear
    yPointsVector = yMin:(yMax-yMin)/(NumberYPoints-1):yMax;

    %Cheby
    %yPointsVector = (((-cos((0:(NumberYPoints-1)) * pi /(NumberYPoints-1))) + 1)/2) *(yMax-yMin) + yMin;
    
    %yPointsVector = yMin + (yMax-yMin)* (-cos((0:(NumberYPoints-1)) * pi /(NumberYPoints-1)) + 1)/2;
    %log
    %yPointsVector = exp(LogyMin:(LogyMax-LogyMin)/(NumberYPoints-1):LogyMax);        
% Set up G handle. x1 is B(t+1), x2 is X(t+1) and X3 is H(t+1)
    GHistory            = zeros(NumberYPoints,economy.HouseStates,BigT);
    PHistory            = zeros(NumberYPoints,economy.HouseStates,3,BigT);
    PHistory_withadj    = PHistory;


    BestPolicyHolder    = zeros(NumberYPoints,economy.HouseStates,3);
    %New G Value for final Period
        %Optimal value of htplus1 from my algebra after adjusting is
    for iJ = 1:economy.HouseStates
        
        Const_b = ((1-lambda-Rf)*economy.HPriceVec(iJ))/Rf;
        Const_f = (-Const_b/gamma)^(1/(alpha-1));
        Const_e =  Const_f /  (1-Const_f*Const_b);        
        htplus1starwithadj                  = Const_e                 * yPointsVector;
        cWithAdjstar                        = (1 + Const_b * Const_e) * yPointsVector;         
        cNoAdjstar                          = lambda*economy.HPriceVec(iJ) + Const_b     + yPointsVector;
        GHistoryNoAdj                       = IntensiveUtilHandle(cNoAdjstar,1);    
        GHistoryWithAdj                     = IntensiveUtilHandle(cWithAdjstar,htplus1starwithadj) ;            
        [GHistory(:,iJ,BigT), BestIndex]       = max([GHistoryWithAdj;GHistoryNoAdj],[],1);
        OneIfAdjBetter                      =  (1 - (BestIndex-1));   
        cstar                               = .999 * (cWithAdjstar .* OneIfAdjBetter + cNoAdjstar .* (1-OneIfAdjBetter));        %Could do this exact with budget const.  
        hstar                               = htplus1starwithadj .* OneIfAdjBetter +  (1-OneIfAdjBetter);
        PHistory_withadj(:,iJ,:,BigT)       = [cstar',0 * yPointsVector', hstar'];
        BestPolicyHolder(:,iJ,:)            = PHistory_withadj(:,iJ,:,BigT);
    end
    economy.extrapvalmax = max(GHistory(:,:,BigT))';
    %CleanG = zeros(NumberYPoints,economy.HouseStates);
    %MVec = zeros(NumberYPoints,economy.HouseStates); %This is replaced later, but allows executing partial code from function. 
    
    fprintf('Most recent loop: %i is done in %f seconds. \n',[BigT,floor(toc)]);
    fprintf('Set model handles\n');    
toc;

%% Check starting point rule
tic;
fprintf('Check Starting Values to Make sure valid\n');
%Note that the constraints are not exactly the same here because this is the
%final period and I'm assuming that in BigT+1 the price is constant from BigT to
%simplify
    for iJ = 1:economy.HouseStates
        economy.CurrentHouseState   = iJ;
     
        economy.CurrentPt           = economy.HPriceVec(iJ);
        PTemp = reshape(PHistory_withadj(:,iJ,:,BigT),[],3);
        cLower_withadj = (PTemp(:,1)                                                                                                >= 0 );
        cUpper_withadj = (PTemp(:,1)                                                                                                <= yPointsVector' );
        xLower_withadj = (PTemp(:,2)                                                                                                >= 0 );
        xUpper_withadj = (PTemp(:,2)                                                                                                <= yPointsVector' * Rf / (Rf - RtDown));
        hLower_withadj = (PTemp(:,3)                                                                                                >= 0 );
        hUpper_withadj = (PTemp(:,3)                                                                                                <= yPointsVector' * Rf / (Rf - 1) );
        %eqn of motion
        %Assumes prices constant in T+1, different conditions later
        MLower_withadj = (PTemp * [-Rf; RtDown-Rf; (1-economy.lambda-economy.Rf ) * economy.CurrentPt] >= -(yPointsVector' + (PTemp(:,3)==1) * economy.lambda * economy.CurrentPt) * Rf);
        MUpper_withadj = (PTemp * [-Rf; RtUp - Rf; (1-economy.lambda-economy.Rf) * economy.CurrentPt]  <= inf );
        TestMatrix = [cLower_withadj , cUpper_withadj , xLower_withadj , xUpper_withadj , hLower_withadj , hUpper_withadj , MLower_withadj , MUpper_withadj];
        TotalWithAdj = sum(TestMatrix,2);
        %sum(TestMatrix,1)
        FracValidStartingWithAdj                                                 = ( sum(TotalWithAdj)) / (8*NumberYPoints);
        fprintf('Valid fraction of intial starting points when price is %f: %f.\n',[economy.HPriceVec(iJ),FracValidStartingWithAdj])
    end


toc;

%% Final initialization for optimization
tic;
    fprintf('Start Discrete Optimization\n');
    economy.FinalT      = 1;         
    NumSinWeights       = size(bn_tune,1);     
    Refine              = RefineGridMax;      
    DiscreteEnd         = 1;
    valuchange          = 0;
    PolChange           = 10;
    
    SpacingVer          = 1 + NumSinWeights; 
    NumSpacings         = SpacingVer;            
    fprintf('Strarting from scratch. Initial points with linear spacing\n');        
    SinPower = 0; %bn_tune(min(NumSpacings,NumSinWeights));
    economy.LinearSpacing = true;
    yPointsCopy = yPointsVector;            
    economy.CurrPlcyResPt = max(min(floor(economy.MaxPlcyResPt/NumSpacings * (NumSpacings-SpacingVer+1)),economy.MaxPlcyResPt),MinPlcyResPt);
    
    fprintf('Starting with %i policy points and gradually increasing to %i points \n',[economy.CurrPlcyResPt,economy.MaxPlcyResPt]);    
toc; 

%% Begin discrete optimization
  
for iT = BigT-1:-1:1
    tic        
    PreviousG = GHistory(:,:,iT+1);

    for iJ = 1:economy.HouseStates

        economy.CurrentTime = iT;
        economy = HouseStateProcessor_110824(economy,iJ);
        
        
        parfor iY = 1:NumberYPoints %Optimize for iteration iT
            %[iY,iJ,iT]
            [GHistory(iY,iJ,iT), PHistory_withadj(iY,iJ,:,iT)] = OptimizeValueSct_110822(yPointsCopy(iY),economy,yPointsVector,economy.LinearSpacing,PreviousG,SinPower,BestPolicyHolder(iY,iJ,:)); 
        end %end iY loop over y values                           
        
        [InvalidPoliciesInd ValidPoliciesInd] = CheckPolicy_110822(yPointsVector',PHistory_withadj(:,iJ,:,iT),economy);
        if numel(ValidPoliciesInd) < InitialYPoints
            fprintf('The following points have solutions that are invalid:\n');
            InvalidPoliciesInd
            fprintf('Overwriting with last valid solution.\n');
            PHistory_withadj(InvalidPoliciesInd,iJ,:,iT) = PHistory_withadj(InvalidPoliciesInd,iJ,:,iT+1);
        end
    end %end iJ loop over current housing state
 
    HowardSteps = ceil(max(max(-log(PolChange/(economy.InitialYPoints * economy.HouseStates) ),0),1));


   %Check if it is time to tighten the grid / use howard's step    
    PolChange = sum(sum(sum(abs(PHistory_withadj(:,:,:,iT) - PHistory_withadj(:,:,:,iT+1)))));     
    
     if PolChange <= 1e-13
        if PolChange == 0 && Refine >= 2 
            Refine = Refine - 2;
        else
            Refine = Refine - 1; %Decrease by one if still very small.
        end
    elseif isnan(valuchange)
        Refine = 0;
    else
        Refine = RefineGridMax; %reset if positive policy change        
     end     
    
    economy.extrapvalmax = max(GHistory(:,:,iT))';
    if HowardSteps > 0
        [~,GHistory(:,:,iT)] = howard_until_converge_110824(yPointsVector,GHistory(:,:,iT),PHistory_withadj(:,:,:,iT),economy,HowardSteps);
        economy.extrapvalmax = max(GHistory(:,:,iT))';
    end      
    
    valuchange = max(max(abs(GHistory(:,:,iT)-GHistory(:,:,iT+1))));      
    temptime             = toc;
    timeelabsed          = temptime + timeelabsed; %Time each loop of iT      
    fprintf('Loop: %i is done in %e seconds. Max(change) of %e. Cumulative Pol change %e\n',[iT,temptime,valuchange,PolChange ]);   
  
    if Refine == 0 && SpacingVer == 1 %Howard to end after last tightening
        DiscreteEnd = iT;
        break;        
    elseif Refine == 0 && SpacingVer > 0 %new and tighter policies using sin method around best guess
            [~,GHistory(:,:,iT)] = howard_until_converge_110824(yPointsVector,GHistory(:,:,iT),PHistory_withadj(:,:,:,iT),economy,-1);
            recalibrate = iT;      
            SpacingVer = SpacingVer-1; 
            economy.CurrPlcyResPt = max(min(floor(economy.MaxPlcyResPt/NumSpacings * (NumSpacings-SpacingVer+1)),economy.MaxPlcyResPt),MinPlcyResPt);
            fprintf('\n\nRecreating policies to be put more points near solutions at Time: %i using tightness parameter %f.\n Now using %i policy points\n',[recalibrate,bn_tune(SpacingVer),economy.CurrPlcyResPt]);
            BestPolicyHolder = PHistory_withadj(:,:,:,iT);
            economy.LinearSpacing = false;
            Refine = RefineGridMax;   
            SinPower = bn_tune(SpacingVer);
    end
       
     
    
end %Time loop end  
    economy.RunTimeSeconds = floor(timeelabsed);
    fprintf('Finished discrete steps at loop %i in %i seconds. Largest change of %e. \n\n',[DiscreteEnd,economy.RunTimeSeconds,valuchange]);


%% Begin continuous optimization stage

fprintf('Start Continuous Optimization\n');
Refine = RefineGridMax;
ContEnd = 1;   %In case it runs until the end and the break loop doesn't set the value. 

if isnan(valuchange)
    %Go to end
else %Run normally
    for iT = DiscreteEnd-1:-1:1
        tic;
        PreviousG       = GHistory(:,:,iT+1);
        PreviousPolicy  = PHistory_withadj(:,:,:,iT+1);
        
        for iJ = 1:economy.HouseStates
            economy.CurrentTime = iT;
            economy = HouseStateProcessor_110824(economy,iJ);
            parfor iY=1:economy.InitialYPoints
                 [GHistory(iY,iJ,iT), PHistory_withadj(iY,iJ,:,iT)] = OptimizeValueSct_continuous_110822(yPointsCopy(iY),economy,yPointsVector,PreviousG,reshape(PreviousPolicy(iY,iJ,:),[1 3]));  
            end 
        end
        %valuchange = max(abs(GHistory(:,:,iT)-GHistory(:,:,iT+1)));
        %PolChange = sum(sum(abs(PHistory_withadj(:,:,iT) - PHistory_withadj(:,:,iT+1))));    
        PolChange = sum(sum(sum(abs(PHistory_withadj(:,:,:,iT) - PHistory_withadj(:,:,:,iT+1)))));     
 
        [InvalidPoliciesInd ValidPoliciesInd] = CheckPolicy_110822(yPointsVector',PHistory_withadj(:,iJ,:,iT),economy);
        if numel(ValidPoliciesInd) < InitialYPoints
            fprintf('The following points have solutions that are invalid:\n');
            InvalidPoliciesInd
            fprintf('Overwriting with last valid solution.\n');
            PHistory_withadj(InvalidPoliciesInd,iJ,:,iT) = PHistory_withadj(InvalidPoliciesInd,iJ,:,iT+1);
        end
        
        economy.extrapvalmax = max(GHistory(:,:,iT))';
        HowardSteps = ceil(max(max(-log(PolChange/(economy.InitialYPoints * economy.HouseStates)),0),1));
        if HowardSteps > 0
            [~,GHistory(:,:,iT)] = howard_until_converge_110824(yPointsVector,GHistory(:,:,iT),PHistory_withadj(:,:,:,iT),economy,HowardSteps);
            economy.extrapvalmax = max(GHistory(:,:,iT))';
        end    
        valuchange = max(max(abs(GHistory(:,:,iT)-GHistory(:,:,iT+1))));    

        if PolChange <= 1e-13
            if PolChange == 0 && Refine >= 2 
                Refine = Refine - 2;
            else
                Refine = Refine -1; %Decrease by one if still very small.
            end
            if Refine == 0
               fprintf('Normal exit from continuous optimization, stoping optimization\n');  
            end
        elseif isnan(valuchange)
            Refine = 0;
            fprintf('Most recent loop had an NAN improvement, stoping optimization\n');   
            
        else
            Refine = RefineGridMax; %reset if positive policy change        
        end      

        if Refine == 0
            ContEnd = iT;
            break; 
        end
        temptime             = toc;
        timeelabsed          = temptime + timeelabsed;    
        fprintf('Most recent loop: %i is done in %e seconds. Largest change of %e. Cumulative Pol change %e\n',[iT,temptime,valuchange,PolChange ]);   
    end
end
economy.RunTimeSeconds = floor(timeelabsed);
fprintf('Finished continuous steps at loop %i in %i seconds. Largest change of %e. \n\n',[ContEnd,economy.RunTimeSeconds,valuchange]);
HowardStart = ContEnd;

%% Howard to end
tic;
fprintf('Starting Howard Steps to end at T: %i\n',HowardStart);
%bestref = sub2ind(size(yUpVec),1:NumberYPoints,BestIndex')';
economy.FinalT = 1;
[~,GHistory(:,:,HowardStart-1)] = howard_until_converge_110824(yPointsVector,GHistory(:,:,HowardStart),PHistory_withadj(:,:,:,HowardStart),economy,-1);
valuchange = max(max(abs(GHistory(:,:,HowardStart-1)-GHistory(:,:,HowardStart))));

%Fill down with policy and value
PHistory_withadj(:,:,:,HowardStart-1:-1:1) = repmat(PHistory_withadj(:,:,:,HowardStart), [1,1,1,size(HowardStart-1:-1:1,2)]);
GHistory(:,:,HowardStart-1:-1:1) = repmat(GHistory(:,:,HowardStart-1), [1,1,size(HowardStart-1:-1:1,2)]);  
    
temptime             = toc;
timeelabsed          = temptime + timeelabsed; 
economy.RunTimeSeconds = floor(timeelabsed);
fprintf('Finished with Howard Steps with final tol %f at time: %i\n',[valuchange,economy.RunTimeSeconds]);

%% Check ending  point rule
tic;

for iJ = 1:economy.HouseStates
        economy = HouseStateProcessor_110824(economy,iJ);        
        [InvalidPoliciesInd ValidPoliciesInd] = CheckPolicy_110822(yPointsVector',PHistory_withadj(:,iJ,:,economy.FinalT),economy);
        if numel(ValidPoliciesInd) < InitialYPoints
            fprintf('The following points have solutions that are invalid of iJ = %i:\n',economy.CurrentHouseState );
            InvalidPoliciesInd
        else
            fprintf('There are no invalid solutions for iJ = %i:\n',economy.CurrentHouseState );
        end    
end
    
temptime             = toc;
timeelabsed          = temptime + timeelabsed; 
economy.RunTimeSeconds = floor(timeelabsed);
fprintf('Total runtime: %i\n',[economy.RunTimeSeconds]);
   
%% Calculate No Adj region and M    
tic;
%Use initial price only to calculate M
Ptemp = reshape(PHistory_withadj(:,StartingHouseIndex,:,economy.FinalT),[],3);
AdjData = [yPointsVector',Ptemp];
AdjPoints = find (AdjData(:,4)~=1);
MVec = GHistory(AdjPoints,3,economy.FinalT)./ (yPointsVector(AdjPoints))'.^economy.a;

meanM = mean(MVec);
stddevM = std(MVec);
economy.M = median(MVec);

FittedValues = economy.M*yPointsVector(AdjPoints)'.^a ;
fprintf('In the Adjust region M was estimated as %f with stddev %f. \n',[meanM,stddevM]); 
CleanG = GHistory(:,economy.FinalT);
CleanG(AdjPoints) = FittedValues;

indexofs1 = find(PHistory_withadj(:,StartingHouseIndex,3,1) == 1, 1 );
indexofs2 = find(PHistory_withadj(:,StartingHouseIndex,3,1) == 1, 1, 'last' );
if isempty(indexofs2)
    indexofs2 = economy.InitialYPoints;
    
end
if isempty(indexofs1)
    indexofs1 = 1;
end

s1 = yPointsVector(indexofs1);
s2 = yPointsVector(indexofs2);
economy.s1 = s1;
economy.s2 = s2;
economy.indexofs1 = indexofs1;
economy.indexofs2 = indexofs2;

iJ = StartingHouseIndex;
economy.CurrentHouseState = iJ;   
economy.CurrentPt = economy.HPriceVec(iJ);
economy.CurrentHousingStates    = 3;
economy.CurrentHUpDownProb      = reshape(economy.HUpDownProb(economy.CurrentHouseState,:),1,1,[]);  
economy.CurrentHPriceVec        = reshape(economy.HPriceVec(economy.CurrentHouseState - 1 : economy.CurrentHouseState + 1), [1 1 economy.CurrentHousingStates]);


CVec            = (yPointsVector'-Ptemp(:,1)+(Ptemp(:,3)==1)*economy.lambda*economy.CurrentPt).* (economy.Rf./Ptemp(:,3));
Cmat            = repmat(CVec,1,economy.StockStates);
XbaseVec        = (Ptemp(:,2)./Ptemp(:,3));
XbaseMat        = repmat(XbaseVec,1,economy.StockStates);
Xmat            = bsxfun(@times, XbaseMat, economy.RVec-economy.Rf);
Expectedytplus1 = yPointsVector' * 0;

for iK = 1: economy.CurrentHousingStates        
    yplus1Mat       =  Cmat + Xmat + (1-economy.lambda)* economy.CurrentHPriceVec(iK)   - economy.Rf * economy.CurrentPt;
    Expectedytplus1 = Expectedytplus1 + economy.HUpDownProb(iK) * (yplus1Mat * economy.PVec');
end    
        
economy.ReturnPoint = median(Expectedytplus1(AdjPoints));

fprintf('Adjust region from y > %f to y < %f. \n',[s1,s2]); 

temptime             = toc;
timeelabsed          = temptime + timeelabsed; 
economy.RunTimeSeconds = floor(timeelabsed);
fprintf('Finished with calculating no adj. region and M at time: %i\n',[economy.RunTimeSeconds]);
%% Check bounds of yMin and yMax
for iJ = 1:economy.HouseStates
    economy = HouseStateProcessor_110824(economy,iJ);
    
    Ptemp           = reshape(PHistory_withadj(:,iJ,:,economy.FinalT),[],3);
    yplus1Mat       = (repmat((yPointsVector'-Ptemp(:,1)+(Ptemp(:,3)==1)*economy.lambda*economy.CurrentPt).* (economy.Rf./Ptemp(:,3)),[1,economy.StockStates,economy.CurrentHousingStates])) ...
                    + (bsxfun(@times, repmat((Ptemp(:,2)./Ptemp(:,3)),[1,economy.StockStates,economy.CurrentHousingStates]), economy.RVec-economy.Rf)   ) ...
                    + (bsxfun(@times, (1-economy.lambda)* ones(size(Ptemp,1),economy.StockStates,economy.CurrentHousingStates),economy.CurrentHPriceVec) - economy.Rf * economy.CurrentPt);
    
    yMinVec(iJ)     = min(min(min(yplus1Mat)));
    yMaxVec(iJ)     = max(max(max(yplus1Mat)));  
end
economy.yPlus1Min = min(yMinVec);
economy.yPlus1Max = max(yMaxVec);

fprintf('Within s-S bounds y_{t+1} ranges from > %f to y < %f. \n',[economy.yPlus1Min,economy.yPlus1Max]); 
if (economy.yPlus1Min > economy.yMin) && (economy.yPlus1Max<economy.yMax)
    fprintf('The choices of economy.yMin and economy.yMax are adaquate to properly handle the s-S bounds\n');
else
    fprintf('Choose different values or economy.yMin and or economy.yMax. The current choices cannot properly handle the s-S bounds\n');
end
%% Trim policies to be a smaller data object to export


Gout = GHistory(:,:,[1,ContEnd,DiscreteEnd, BigT]);
Pout = PHistory_withadj(:,:,:,[1,ContEnd,DiscreteEnd, BigT]);
