function economy = HouseStateProcessor_110824(economy,iJ)
economy.CurrentHouseState = iJ;
economy.CurrentPt = economy.HPriceVec(iJ);
%{
fprintf('HTransProbStates \n');
economy.HTransProbStates
fprintf('economy.CurrentPt %i\n',economy.CurrentPt);
fprintf('economy.CurrentHouseState %i\n',economy.CurrentHouseState);
%}

if economy.HTransProbStates == 3
    %{
        if economy.CurrentHouseState == 1
            economy.minstate                = economy.CurrentHouseState;
            economy.maxstate                = economy.CurrentHouseState + 1;  
            economy.CurrentHousingStates    = economy.maxstate  - economy.minstate + 1;            
            economy.CurrentHUpDownProb      = reshape(economy.HTransProb(economy.CurrentHouseState,economy.minstate:economy.maxstate),1,1,[]); 
            economy.CurrentHPriceVec        = reshape(economy.HPriceVec(economy.minstate  : economy.maxstate), 1,1, []);

            
        elseif economy.CurrentHouseState == economy.HouseStates
            economy.minstate                = economy.CurrentHouseState - 1;
            economy.maxstate                = economy.CurrentHouseState;     
            economy.CurrentHousingStates    = economy.maxstate  - economy.minstate + 1;            
            economy.CurrentHUpDownProb      = reshape(economy.HTransProb(economy.CurrentHouseState,economy.minstate:economy.maxstate),1,1,[]); 
            economy.CurrentHPriceVec        = reshape(economy.HPriceVec(economy.minstate  : economy.maxstate), 1,1, []);            
        else
            economy.minstate                = economy.CurrentHouseState - 1;
            economy.maxstate                = economy.CurrentHouseState + 1;   
            economy.CurrentHousingStates    = economy.maxstate  - economy.minstate + 1;            
            economy.CurrentHUpDownProb      = reshape(economy.HTransProb(economy.CurrentHouseState,economy.minstate:economy.maxstate),1,1,[]); 
            economy.CurrentHPriceVec        = reshape(economy.HPriceVec(economy.minstate  : economy.maxstate), 1,1, []);            
        end
    %}
    economy.minstate                = max(economy.CurrentHouseState - 1,1);
    economy.maxstate                = min(economy.CurrentHouseState + 1,economy.HouseStates);
    economy.CurrentHousingStates    = economy.maxstate  - economy.minstate + 1;  
    economy.CurrentHPriceVec        = reshape(economy.HPriceVec(economy.minstate  : economy.maxstate), 1,1, []);        
    economy.CurrentHUpDownProb      = reshape(economy.HTransProb(economy.CurrentHouseState,economy.minstate:economy.maxstate),1,1,[]); 
    
    economy.CurrentWeightMat = reshape(bsxfun(@times,repmat(economy.PVec,economy.CurrentHousingStates,1),economy.CurrentHUpDownProb(:))' ,1, economy.StockStates, []);
    
elseif  economy.HTransProbStates == 5        
    economy.minstate                = max(economy.CurrentHouseState - 2,1);
    economy.maxstate                = min(economy.CurrentHouseState + 2,economy.HouseStates);
    economy.CurrentHousingStates    = economy.maxstate  - economy.minstate + 1;  
    economy.CurrentHPriceVec        = reshape(economy.HPriceVec(economy.minstate  : economy.maxstate), 1,1, []);        
    economy.CurrentHUpDownProb      = reshape(economy.HTransProb(economy.CurrentHouseState,economy.minstate:economy.maxstate),1,1,[]); 
        
    economy.CurrentWeightMat = reshape(bsxfun(@times,repmat(economy.PVec,economy.CurrentHousingStates,1),economy.CurrentHUpDownProb(:))' ,1, economy.StockStates, []);
    
else
    fprintf('Danger, wrong number of transition probabilities\n');
end
%{
fprintf('CurrentHUpDownProb \n');
economy.CurrentHUpDownProb
fprintf('CurrentHPriceVec \n');
economy.CurrentHPriceVec
fprintf('CurrentHousingStates \n');
economy.CurrentHousingStates
%}
%