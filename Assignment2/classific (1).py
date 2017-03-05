import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import naive_bayes
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import cluster

df = pd.read_csv('/Users/lgnonato/Meusdocs/Cursos/CUSP-GX-5006/Data/manhattan-dof.csv',index_col=False,delimiter=';')
df_x = df.ix[:,2:]
df_y = df.ix[:,1]

X = df_x.as_matrix()
Y = df_y.as_matrix()

# --------------------
# Normalizing data
# --------------------
Xtmax = np.amax(X,axis=0)
X = np.divide(X,Xtmax)

# --------------------
# Clustering
# --------------------

## Kmeans
#km = cluster.KMeans(n_clusters=2, random_state=0)
#km.fit(X)
#cl = km.labels_

# Hierarchical
#hc = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
#hc.fit(X)
#cl = hc.labels_
#
#plt.scatter(X[:,3],X[:,2],c=cl)

# --------------------
# K-fold CV
# --------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=3)
ts, = Y_test.shape

# --------------------
# Naive Bayes
# --------------------
gnb = naive_bayes.GaussianNB()
gnb.fit(X_train,Y_train)
ypred_gnb = gnb.predict(X_test)
e_gnb = np.sum((ypred_gnb[i] != Y_test[i]) for i in range(0,ts))
print("----------Naive Bayes----------")
print(e_gnb, "misclassified data out of", ts, "(",e_gnb/ts,"%)")

# --------------------
# SVM
# --------------------
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train,Y_train)
ypred_svm_linear = svm_linear.predict(X_test)
e_svm_linear = np.sum((ypred_svm_linear[i] != Y_test[i]) for i in range(0,ts))
print("----------SVM Linear----------")
print(e_svm_linear, "misclassified data out of", ts, "(",e_svm_linear/ts,"%)")

#class_weight={0:10} give a larger importance to class 0
#svm_rbf = svm.SVC(kernel='rbf', gamma=500, class_weight={0:10})
svm_rbf = svm.SVC(kernel='rbf', gamma=500)
svm_rbf.fit(X_train,Y_train)
ypred_svm_rbf = svm_rbf.predict(X_test)
e_svm_rbf = np.sum((ypred_svm_rbf[i] != Y_test[i]) for i in range(0,ts))
print("----------SVM RBF----------")
print("number of support vectors",len(svm_rbf.support_))
print(e_svm_rbf, "misclassified data out of", ts, "(",e_svm_rbf/ts,"%)")


# --------------------
# SVM_RBF (analyzing gamma)
# --------------------
#l=[]
#for g in range(1,500,2):
#    svm_rbf = svm.SVC(kernel='rbf', gamma=float(g))
#    svm_rbf.fit(X_train,Y_train)
#    ypred_svm_rbf = svm_rbf.predict(X_test)
#    e_svm_rbf = np.sum((ypred_svm_rbf[i] != Y_test[i]) for i in range(0,ts))
#    l.append(e_svm_rbf/ts)
#
#plt.plot(l)

plt.subplot(1,3,1)
plt.scatter(X_test[:,2],X_test[:,3],c=ypred_gnb)
plt.subplot(1,3,2)
plt.scatter(X_test[:,2],X_test[:,3],c=ypred_svm_linear)
plt.subplot(1,3,3)
plt.scatter(X_test[:,2],X_test[:,3],c=ypred_svm_rbf)
#svm_rbf_sv = svm_rbf.support_
#plt.scatter(X[svm_rbf_sv,2],X[svm_rbf_sv,3],color="red")
plt.show()
