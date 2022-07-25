import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sn
from sklearn.metrics import confusion_matrix, roc_curve
import sklearn.neural_network as nn
import sklearn.svm as sv
from sklearn.metrics import roc_auc_score

TrainDT = pd.read_csv('./fake-news/train.csv', header=None)
TrainDT = TrainDT.iloc[1:, 3:5]
col = ['text', 'label']
TrainDT.columns = col
TrainDT = TrainDT[pd.notnull(TrainDT['text'])]
X_train = TrainDT['text']
Y_train = TrainDT['label'].astype('int')

TestDT = pd.read_csv('./fake-news/test.csv', header=None)
TestDT = TestDT.iloc[1:, 3:4]
col = ['text']
TestDT.columns = col
TestDT2 = TestDT[pd.notnull(TestDT['text'])]
X_test = TestDT2['text']

SubmitDT = pd.read_csv('./fake-news/submit.csv', header=None)
SubmitDT = SubmitDT.iloc[1:, 1:2]
col = ['label']
SubmitDT.columns = col
SubmitDT = SubmitDT[pd.notnull(TestDT['text'])]
Y_test = SubmitDT['label'].astype('int')

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
Train_labels = Y_train

X_test_tfidf = count_vect.transform(X_test)

NN = nn.MLPClassifier(hidden_layer_sizes=(15, 20), activation='relu', solver='adam', max_iter=300, alpha=0.001)
NN.fit(X_train_tfidf, Train_labels)
trAcc = NN.score(X_train_tfidf, Train_labels)

Clsfr = sv.SVC(kernel = 'rbf')
Clsfr.fit(X_train_tfidf, Train_labels)
trAcc=Clsfr.score(X_train_tfidf, Train_labels)
teAcc=Clsfr.score(X_test_tfidf, Y_test)

print('train Acc SVM = ' , trAcc )
print('test Acc SVM = ' , teAcc )

y_pred_MLP = NN.predict(X_test_tfidf)
y_pred_SVM = Clsfr.predict(X_test_tfidf)

############ MLP ##################
conf_mat = confusion_matrix(Y_test, y_pred_MLP)
# constant for classes
classes = ('0', '1')
TP = conf_mat[0][0]
FP = conf_mat[1][0]
FN = conf_mat[0][1]
TN = conf_mat[1][1]

# Calculate The Accuracy Using The Confusion Matrix
Accuracy = 100 * (TP + TN) / (TP + TN + FN + FP)
print('Calculate The Accuracy : %.4f %%' % Accuracy)
# Calculate The Error rate Using The Confusion Matrix
ErrorRate = 100 * (FP + FN) / (TP + TN + FN + FP)
print('Calculate The Error Rate : %.4f %%' % ErrorRate)
# Calculate The Precision Using The Confusion Matrix
Precision = 100 * TP / (TP + FP)
print('Calculate The Precision : %.4f %%' % Precision)
# Calculate The Recall Using The Confusion Matrix
Recall = 100 * TP / (TP + FN)
print('Calculate The Recall : %.4f %%' % Recall)
# Calculate The F1 Score Using The Confusion Matrix
F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
print('Calculate The F1_Score : %.4f %%' % F1_Score)
# Calculate The FPR Using The Confusion Matrix
FPR = 100 * FP / (TN + FP)
print('Calculate The FPR : %.4f %%' % FPR)
# Calculate The FPR Using The Confusion Matrix
Specificity = 100 * TN / (TN + FP)
print('Calculate The Specificity(True Negative Rate) : %.4f %%' % Specificity)

df_cm = pd.DataFrame(conf_mat / np.sum(conf_mat) * 100, index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(6, 3))
sn.heatmap(df_cm, annot=True)
plt.show()

auc = roc_auc_score(Y_test, y_pred_MLP) * 100
print('Calculate The ROC AUC Score : %.4f %%' % auc)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_MLP)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.show()

############ SVM ##################

conf_mat1 = confusion_matrix(Y_test, y_pred_SVM)
# constant for classes
classes = ('0', '1')
TP = conf_mat1[0][0]
FP = conf_mat1[1][0]
FN = conf_mat1[0][1]
TN = conf_mat1[1][1]

# Calculate The Accuracy Using The Confusion Matrix
Accuracy = 100 * (TP + TN) / (TP + TN + FN + FP)
print('Calculate The Accuracy : %.4f %%' % Accuracy)
# Calculate The Error rate Using The Confusion Matrix
ErrorRate = 100 * (FP + FN) / (TP + TN + FN + FP)
print('Calculate The Error Rate : %.4f %%' % ErrorRate)
# Calculate The Precision Using The Confusion Matrix
Precision = 100 * TP / (TP + FP)
print('Calculate The Precision : %.4f %%' % Precision)
# Calculate The Recall Using The Confusion Matrix
Recall = 100 * TP / (TP + FN)
print('Calculate The Recall : %.4f %%' % Recall)
# Calculate The F1 Score Using The Confusion Matrix
F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
print('Calculate The F1_Score : %.4f %%' % F1_Score)
# Calculate The FPR Using The Confusion Matrix
FPR = 100 * FP / (TN + FP)
print('Calculate The FPR : %.4f %%' % FPR)
# Calculate The FPR Using The Confusion Matrix
Specificity = 100 * TN / (TN + FP)
print('Calculate The Specificity(True Negative Rate) : %.4f %%' % Specificity)

df_cm = pd.DataFrame(conf_mat1 / np.sum(conf_mat1) * 100, index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(6, 3))
sn.heatmap(df_cm, annot=True)
plt.show()

auc = roc_auc_score(Y_test, y_pred_SVM) * 100
print('Calculate The ROC AUC Score : %.4f %%' % auc)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_SVM)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.show()
