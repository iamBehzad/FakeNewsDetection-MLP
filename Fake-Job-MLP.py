import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sn
from sklearn.metrics import confusion_matrix, roc_curve
import sklearn.neural_network as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

############  Read Data ##################
DT = pd.read_csv('./fake_job_postings/fake_job_postings.csv', header=None)
DT = DT.iloc[1:, [6, 17]]
col = ['text', 'label']
DT.columns = col

############  Preprocessing Data ##################
DT = DT[pd.notnull(DT['text'])]

############  Split Data ##################
X_train, X_test, Y_train, Y_test = train_test_split(DT['text'], DT['label'], test_size=0.33, random_state=42)

############  Feature Extraction (TD-IDF) ##################
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
Train_labels = Y_train

X_test_tfidf = count_vect.transform(X_test)

############ Create MLP Model & Train ##################
NN = nn.MLPClassifier(hidden_layer_sizes=(15, 20), activation='relu', solver='adam', max_iter=300, alpha=0.001)
NN.fit(X_train_tfidf, Train_labels)
trAcc = NN.score(X_train_tfidf, Train_labels)

print('train Acc MLP = ' , trAcc )

############ Test Model ##################
y_pred_MLP = NN.predict(X_test_tfidf)
Y_test = [int(i) for i in Y_test]
y_pred_MLP = [int(i) for i in y_pred_MLP]

############ Confusion Matrix ##################
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

############ ROC_AUC & ROC_Curve ##################
auc = roc_auc_score(Y_test, y_pred_MLP)
print('Calculate The ROC AUC Score : %.4f %%' % auc)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_MLP)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.show()
