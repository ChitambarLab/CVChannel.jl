using LinearAlgebra
using CVChannel
using Convex

pVal = 0.0; qVal = 0.0;
pStep = 0.05; qStep = 0.05;
pCtr = 0; qCtr = 0;
pqArray = zeros(11,11);
cvArray = zeros(11,11);
cv2Array = zeros(11,11);
while pVal < 1.01
     qVal = 0;
     while qVal < 1.01
         #Get Data
         dephrasurepq(ρ) = dephrasureChannel(ρ,pVal,qVal)
         origChoi = choi(dephrasurepq,2);
         [s.val1,s.Opt1] = minEntropyPPTDual(origChoi,2,3);
         parallelChoi = PermuteSystems(kron(origChoi,origChoi),[1,3,2,4],[2,3,2,3]);
         [s.val2,s.Opt2] = minEntropyPPTPrimal(parallelChoi,4,9);

         #Store Data
         cvArray(pCtr+1,qCtr+1) = s.val1;
         cv2Array(pCtr+1,qCtr+1) = s.val2;
         data{pCtr+1,qCtr+1} = s;

         #Iterate
         qCtr = qCtr + 1;
         qVal = qCtr*qStep;
     end
     #Iterate
     pCtr = pCtr + 1;
     pVal = pCtr*pStep;
     qVal = 0; qCtr = 0;
end




#This is to plot the results if you would like
#(I have package add to avoid dependencies if you don't need plots)
Pkg.add("Plots")
Pkg.add("LaTeXStrings")
using Plots
using LaTeXStrings
x = [0:0.01:0.3;];
y = results[:,2:4];
titleStr = "Multiplicativity of Holevo-Werner Channel";
labelStr = [L"cv(\mathcal{N})" L"cv \left(\mathcal{N}^{\otimes 2} \right)" L"cv \left(\mathcal{N}^{\otimes 2} \right) - cv(\mathcal{N})"];
plot(x,
     y,
     xlims = (0,0.31),
     title = titleStr,
     label= labelStr,
     lw=1)
