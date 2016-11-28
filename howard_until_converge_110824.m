function [yPointsVector,OutputG] = howard_until_converge_110824(yPointsVector,InputG,PolicyHoward,economy,howardsteps)
%Run howard step on policy until convergence, NAN, or designated # of times 
%Use a negative number to run until convergence


NewG   = InputG;
OldG    = NewG;
%yMin    = yPointsVector(1);
%yMax    = yPointsVector(end);
tol     = 1;

MaxSteps = howardsteps;
if (howardsteps <  0) || (howardsteps == inf) 
    MaxSteps = inf;
    howardsteps = 1000;    
end
TotalSteps = 0;

while (tol > 0) && isnan(tol) == 0 && (howardsteps > TotalSteps)
    extrapvalmax = max(NewG);
    for iJ = 1:economy.HouseStates
        economy = HouseStateProcessor_110824(economy,iJ);


        CVec            = (yPointsVector'-PolicyHoward(:,iJ,1)+(PolicyHoward(:,iJ,3)==1)*economy.lambda*economy.CurrentPt).* (economy.Rf./PolicyHoward(:,iJ,3));
        Cmat            = repmat(CVec,1,economy.StockStates);
        XbaseVec        = (PolicyHoward(:,iJ,2)./PolicyHoward(:,iJ,3));
        XbaseMat        = repmat(XbaseVec,1,economy.StockStates);
        Xmat            = bsxfun(@times, XbaseMat, economy.RVec-economy.Rf);

        Expvplus1Mat        = zeros(economy.InitialYPoints,1);
        for iK = economy.minstate: economy.maxstate         
            yplus1Mat       =  Cmat + Xmat + (1-economy.lambda)* economy.CurrentHPriceVec(iK-economy.minstate+1)   - economy.Rf * economy.CurrentPt;                               
            vplus1Mat       = beninterp1_for_mat(yPointsVector',OldG(:,iK),yplus1Mat,economy.extrapvalmin,economy.yMin,extrapvalmax(iK),economy.yMax);
            Expvplus1Mat    = Expvplus1Mat + economy.CurrentHUpDownProb(iK-economy.minstate+1) * sum(bsxfun(@times, vplus1Mat , economy.PVec),2); 
        end        
        NewG(:,iJ) = (1/(1-economy.rho)).*(economy.gamma*PolicyHoward(:,iJ,3).^economy.alpha+PolicyHoward(:,iJ,1).^economy.alpha).^((1-economy.rho)/economy.alpha) ...
                    + PolicyHoward(:,iJ,3).^(1-economy.rho) * economy.beta .* (Expvplus1Mat);



    end
    tol1 = max(abs(NewG-OldG));
    tol = max(tol1);         
    
    OldG = NewG;

    TotalSteps = TotalSteps + 1;    
    
end        

fprintf('Running %i Howard steps of a maximum %i \n',[TotalSteps,MaxSteps])
OutputG = NewG;        
   


