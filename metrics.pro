; IDL FUNCTION TO COMPUTE THE PERFORMANCE METRICS FROM A CONTIGENCY TABLE

PRO metrics,TP,TN,FP,FN

m=TP+FP+TN+FN
recall=TP/(TP+FN)
precision=TP/(TP+FP)
recalln=TN/(TN+FP)
precisionn=TN/(TN+FN)
accuracy=(TP+TN)/(TP+TN+FP+FN)
f1=2.0*recall*precision/(recall+precision)
f1n=2.0*recalln*precisionn/(recalln+precisionn)
p=tp+fn
n=tn+fp
HSS1=(tp+tn-n)/p
EE=((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/m
HSS2=(tp+tn-EE)/(tp+fp+fn+tn-EE)
CH=(tp+fp)*(tp+fn)/m
GS=(tp-CH)/(tp+fp+fn-CH)
TSS=tp/(tp+fn)-fp/(fp+tn)

print,'accuracy=',accuracy
print,'precision=',precision
print,'recall=',recall
print,'precision (negative)=',precisionn
print,'recall (negative)=',recalln
print,'f1-score=',f1
print,'f1-score (negative)=',f1n
print,'HSS1=',HSS1
print,'HSS2=',HSS2
print,'GS=',GS
print,'TSS=',TSS

END
