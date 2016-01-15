import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import re	
import sys

pred_file = sys.argv[1]
test_file = sys.argv[2]

f = open(pred_file,"r")

y_prob = list()
y_prob_neg = list()
y_p = list()
f.readline()
for i,prob in enumerate(f):
    p = prob.split(" ")
    y_prob.append(float(p[1]))
    y_prob_neg.append(float(p[2]))
    y_p.append(int(p[0]))
f.close()   
y_scores = np.array(y_prob)
y_pred = np.array(y_p)

probs = zip(y_prob,y_prob_neg)

f = open(test_file)

y_t = list()
for y in f:
	if ((y[0] == '0') or (y[0] == '-')):	
		y_t.append(0)
	elif (y[0] == '1'):
	
		y_t.append(int(y[0]))

f.close()

y_true = np.array(y_t)


rss = 0
for label,prob in zip(y_t,probs):
    if (label == 1):
        rss = rss + ((1 - prob[0])**2)
    else:
        rss = rss + ((1 - prob[1])**2) 

print "Accuracy = ",accuracy_score(y_true,y_pred)
print "F-score = ",f1_score(y_true,y_pred)
print "Average RSS = ",rss/len(y_t)
print "ROC_AUC_score =",roc_auc_score(y_true,y_scores)
print "Average Precision = ",average_precision_score(y_true,y_scores)
