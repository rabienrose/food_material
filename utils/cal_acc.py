TP=114
TN=2310
FP=24
FN=77
recall = TP/(FN+TP)
precision=TP/(TP+FP)
accuracy=(TP+TN)/(TP+TN+FP+FN)
pre_positive=TP+FP
pre_negative=TN+FN

print("recall: %f" % recall)
print("precision: %f" % precision)
print("accuracy: %f" % accuracy)
print("pre_positive: %f" % pre_positive)
print("pre_negative: %f" % pre_negative)