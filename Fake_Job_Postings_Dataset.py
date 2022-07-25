import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sn
from sklearn.metrics import confusion_matrix
import sklearn.neural_network as nn

TrainDT = pd.read_csv('./liar_dataset/train.tsv', sep='\t' , header=None)

TrainDT = TrainDT.iloc[:, 1:3]
col = ['label', 'text']
TrainDT.columns = col
TrainDT = TrainDT[pd.notnull(TrainDT['text'])]

TrainDT.loc[TrainDT['label'] == 'false', 'label'] = 0
TrainDT.loc[TrainDT['label'] == 'pants-fire', 'label'] = 1
TrainDT.loc[TrainDT['label'] == 'barely-true', 'label'] = 2
TrainDT.loc[TrainDT['label'] == 'half-true', 'label'] = 3
TrainDT.loc[TrainDT['label'] == 'mostly-true', 'label'] = 4
TrainDT.loc[TrainDT['label'] == 'true', 'label'] = 5

X_train = TrainDT['text']
Y_train = TrainDT['label'].astype('int')

TestDT = pd.read_csv('./liar_dataset/test.tsv', sep='\t' , header=None)

TestDT = TestDT.iloc[:, 1:3]
col = ['label', 'text']
TestDT.columns = col
TestDT = TestDT[pd.notnull(TestDT['text'])]

TestDT.loc[TestDT['label'] == 'false', 'label'] = 0
TestDT.loc[TestDT['label'] == 'pants-fire', 'label'] = 1
TestDT.loc[TestDT['label'] == 'barely-true', 'label'] = 2
TestDT.loc[TestDT['label'] == 'half-true', 'label'] = 3
TestDT.loc[TestDT['label'] == 'mostly-true', 'label'] = 4
TestDT.loc[TestDT['label'] == 'true', 'label'] = 5

X_test = TestDT['text']
Y_test = TestDT['label'].astype('int')

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
Train_labels = Y_train

NN = nn.MLPClassifier(hidden_layer_sizes = (15, 20), activation = 'relu', solver = 'adam', max_iter = 300, alpha = 0.001)
NN.fit(X_train_tfidf, Train_labels)
trAcc=NN.score(X_train_tfidf, Train_labels)

print(trAcc)

X_test_tfidf = count_vect.transform(X_test)
y_pred = NN.predict(X_test_tfidf)
conf_mat = confusion_matrix(Y_test, y_pred)

# constant for classes
classes = ('0', '1', '2', '3', '4', '5')

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