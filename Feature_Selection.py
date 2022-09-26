import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OneHotEncoder


BCdata = pd.read_csv('DATA.csv', sep=';')
BCdata.head()
BCdata.info()

#encoding the catigorical feature

  #encoding by dummy
#one_hot = pd.get_dummies(BCdata.diagnosis)
#one_hot=one_hot.drop('B', axis='columns')

data=BCdata['diagnosis'].values
new_df = pd.DataFrame(data, columns = ['diagnosis']) #creat dataframe with the catigorical feature
new_df.head()

le = preprocessing.LabelEncoder() #pass new dataframe to the encoder
new_df = new_df.apply(le.fit_transform)
new_df.head()
 
BCdata=BCdata.drop('diagnosis', axis='columns') #replace catigorical data with numrical data
BCdata=BCdata.join(new_df)

x = BCdata.drop('diagnosis', axis='columns') 
y = BCdata['diagnosis'].values

####################################################################
###################### Filter technique ############################
####################################################################
x.drop(["id"],axis=1,inplace=True)
x.head()

  #get variance of each feature
col_names=x.columns
for i in col_names:
    print(i, ": ", x[i].var())
    
  #selecting best subset by threshold
      #if variance of feature less than threshold then we ignore it
constant_feature_selector = VarianceThreshold(threshold=0.001)
constant_feature_selector.fit(x)
constant_feature_selector.get_support() #show which feature was selected

best_sub=x.columns[constant_feature_selector.get_support()]
X=x[best_sub]
X.head()

  #split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

                      #**learning algorithms**#
#using K-Fold Cross-Validation = 10
#first: RandomForestClassifier

r_forest = RandomForestClassifier(max_depth=100, random_state=0) #classifer
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(r_forest, X, y, scoring='accuracy', cv=cv) #accuracy without split
rf_score = np.mean(scores)
r_forest = r_forest.fit(X_train,y_train) #fit classifer
y_pred = r_forest.predict(X_test) #Y predict
print('Accuracy : %.3f' % (rf_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

#second: KNeighborsClassifier
 for i in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=i)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    scores = cross_val_score(neigh, X, y, scoring='accuracy', cv=cv)
    print('k=%d : %.3f' % (i, (np.mean(scores))))
  #best k parameter is 7
neigh = KNeighborsClassifier(n_neighbors=7) #classifer
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(neigh, X, y, scoring='accuracy', cv=cv) #accuracy without split
knn_score = np.mean(scores)
neigh = neigh.fit(X_train,y_train) #fit classifer
y_pred = neigh.predict(X_test) #Y predict
print('Accuracy : %.3f' % (knn_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

#third: DecisionTree
clf = DecisionTreeClassifier() #classifer
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv) #accuracy without split
clf_score = np.mean(scores)
clf = clf.fit(X_train,y_train) #fit classifer
y_pred = clf.predict(X_test) #Y predict
print('Accuracy : %.3f' % (clf_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

####################################################################
######################## Embedded technique ########################
###################################################################
  #->Note for me: i can split data and pass to LG X_train, y_train to get best features then
                                #split data again with new X to try classifiers
#lasso regression
  #selecting the best subset using logistic regression with penalty l1
selection = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
selection.fit(x, y)
  #the selected features
selected_features = x.columns[(selection.get_support())]
X=x[selected_features]
  #see the deleted features
removed_features = x.columns[(selection.estimator_.coef_ == 0).ravel().tolist()]

  #split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

                        #**learning algorithms**#
#using K-Fold Cross-Validation = 10
#first: RandomForestClassifier
r_forest = RandomForestClassifier(max_depth=100, random_state=0) #classifer 
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(r_forest, X, y, scoring='accuracy', cv=cv) #accuracy without split
rf_score = np.mean(scores)
r_forest = r_forest.fit(X_train,y_train) #fit classifer
y_pred = r_forest.predict(X_test) #Y predict

print('Accuracy : %.3f' % (rf_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

#second: KNeighborsClassifier
 for i in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=i)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    scores = cross_val_score(neigh, X, y, scoring='accuracy', cv=cv)
    print('k=%d : %.3f' % (i, (np.mean(scores))))
  #best k parameter is 7
neigh = KNeighborsClassifier(n_neighbors=7) #classifer
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(neigh, X, y, scoring='accuracy', cv=cv) #accuracy without split
knn_score = np.mean(scores)
neigh = neigh.fit(X_train,y_train) #fit classifer
y_pred = neigh.predict(X_test) #Y predict
print('Accuracy : %.3f' % (knn_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

#third: DecisionTree
clf = DecisionTreeClassifier() #classifer
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv) #accuracy without split
clf_score = np.mean(scores)
clf = clf.fit(X_train,y_train) #fit classifer
y_pred = clf.predict(X_test) #Y predict
print('Accuracy : %.3f' % (clf_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

####################################################################
######################## Wrapper technique #########################
####################################################################
  #->Note for me: we insalled mlxtend first
#Backword feature selection
    #split data 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
Ir=LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=42, n_jobs=-1, max_iter=500)
Ir.fit(X_train,y_train)
  #bfs function take time in running
bfs=SequentialFeatureSelector(Ir, k_features='best', forward=False, n_jobs=-1)
bfs.fit(X_train,y_train)
selected_features=list(bfs.k_feature_names_)
X=x[selected_features]
X.head()
  #split data with best features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

                   #**learning algorithms**#
#using K-Fold Cross-Validation = 10
#first: RandomForestClassifier
r_forest = RandomForestClassifier(max_depth=100, random_state=0) #classifer 
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(r_forest, X, y, scoring='accuracy', cv=cv) #accuracy without split
rf_score = np.mean(scores)
r_forest = r_forest.fit(X_train,y_train) #fit classifer
y_pred = r_forest.predict(X_test) #Y predict

print('Accuracy : %.3f' % (rf_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

#second: KNeighborsClassifier
 for i in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=i)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    scores = cross_val_score(neigh, X, y, scoring='accuracy', cv=cv)
    print('k=%d : %.3f' % (i, (np.mean(scores))))
  #best k parameter is 6
neigh = KNeighborsClassifier(n_neighbors=6) #classifer
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(neigh, X, y, scoring='accuracy', cv=cv) #accuracy without split
knn_score = np.mean(scores)
neigh = neigh.fit(X_train,y_train) #fit classifer
y_pred = neigh.predict(X_test) #Y predict
print('Accuracy : %.3f' % (knn_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

#third: DecisionTree
clf = DecisionTreeClassifier() #classifer
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) #K-fold
scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv) #accuracy without split
clf_score = np.mean(scores)
clf = clf.fit(X_train,y_train) #fit classifer
y_pred = clf.predict(X_test) #Y predict
print('Accuracy : %.3f' % (clf_score))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))