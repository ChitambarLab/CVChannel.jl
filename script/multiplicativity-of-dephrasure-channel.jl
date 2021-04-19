using LinearAlgebra
using CVChannel
using Convex

pVal = 0.0; qVal = 0.0;
pStep = 0.1; qStep = 0.1;
pCtr = 0; qCtr = 0;
pqArray = zeros(11,11);
cvArray = zeros(11,11);
cv2Array = zeros(11,11);
while pVal < 1.01
     qVal = 0;
     while qVal < 1.01
         #Get Data
         dephrasurepq(ρ) = dephrasureChannel(ρ,pVal,qVal)
         origChoi = choi(dephrasurepq,2,3);
         val1, Opt1 = minEntropyPPTDual(origChoi,2,3);
         parallelChoi = permuteSubsystems(kron(origChoi,origChoi),[1,3,2,4],[2,3,2,3]);
         val2, Opt2 = minEntropyPPTPrimal(parallelChoi,4,9);

         #Store Data
         cvArray[pCtr+1,qCtr+1] = val1;
         cv2Array[pCtr+1,qCtr+1] = val2;

         #Iterate
         qCtr = qCtr + 1;
         qVal = qCtr*qStep;
     end
     #Iterate
     pCtr = pCtr + 1;
     pVal = pCtr*pStep;
     qVal = 0; qCtr = 0;
end

diffArray = cv2Array - cvArray.^2;
"Checking diffArray shows this is non-multiplicative"
end
