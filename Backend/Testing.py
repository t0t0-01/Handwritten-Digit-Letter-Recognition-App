import time 
import joblib
import numpy as np 
from Knn import KNN 
from sklearn import svm
from sklearn import metrics
from collections import Counter
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 

def test(x_train, y_train, x_test, y_test, fit=False):
    """ 
    Our KNN 
    """  
    if fit:
        knn_model = KNN(5)
        knn_model.fit(x_train, y_train) 
        joblib.dump(knn_model, './models/knn_model')
    else:
        knn_model = joblib.load('./models/knn_model')
    t1 = time.time()
    res1 = []
    for i in x_test:
        a = knn_model.predict(i)
        b = Counter(a)
        res1.append(b.most_common()[0][0])

    print('Our KNN accuracy: ', metrics.accuracy_score(y_test, res1))
    print (time.time()-t1)
    
    """
    Sci-kit KNN
    """ 
    if fit:
        sl_knn_model = KNeighborsClassifier(n_neighbors=5)
        sl_knn_model = sl_knn_model.fit(x_train, y_train) 
        joblib.dump(sl_knn_model, './models/sl_knn_model')
    else:
        sl_knn_model = joblib.load('./models/sl_knn_model')
    t2 = time.time()
    res2 = sl_knn_model.predict(x_test)

    print('SciKit KNN accuracy: ', metrics.accuracy_score(y_test.reshape(-1,1), res2))
    print (time.time()-t2)


    """
    Sci-Kit SVM
    """
    if fit:
        svm_model = svm.NuSVC()
        svm_model.fit(x_train, y_train)
        joblib.dump(svm_model, './models/svm_model')
    else:
        svm_model = joblib.load('./models/svm_model')
    t3 = time.time()
    res3 = svm_model.predict(x_test)
        
    print('SciKit SVM accuracy: ', metrics.accuracy_score(y_test.reshape(-1,1), res3))
    print (time.time()-t3)


    """
    Decision Tree
    """
    if fit:
        tree_model = DecisionTreeClassifier()
        tree_model.fit(x_train, y_train) 
        joblib.dump(tree_model, './models/tree_model')
    else:
        tree_model = joblib.load('./models/tree_model')
    t4 = time.time()
    res4 = tree_model.predict(x_test)
    print("Decision Tree Accuracy: ", metrics.accuracy_score(y_test.reshape(-1,1), res4))
    print(time.time() - t4)

    """
    Random Forest
    """
    if fit:
        rf_model = RandomForestClassifier(n_estimators=150)
        rf_model.fit(x_train, y_train) 
        joblib.dump(rf_model, './models/rf_model')
    else:
        rf_model = joblib.load('./models/rf_model')
    t5 = time.time()
    res5 = rf_model.predict(x_test)
    print("Random Forest Accuracy: ", metrics.accuracy_score(y_test.reshape(-1,1), res5))
    print(time.time() - t5)
    

    """
    Ensemble of Models
    """
    if fit:
        models = [('rf', RandomForestClassifier(n_estimators=150)), ('knn', KNeighborsClassifier(n_neighbors=5)) , ('svm', svm.NuSVC())]
        ensemble = VotingClassifier(estimators=models, voting='hard')
        ensemble = ensemble.fit(x_train, y_train)
        joblib.dump(ensemble, './models/ensemble_model')
    else:
        ensemble = joblib.load('./models/ensemble_model')
    t7 = time.time()
    res7 = ensemble.predict(x_test) 
    print("Ensemble Accuracy: ", metrics.accuracy_score(y_test.reshape(-1,1), res7))
    print(time.time() - t7)
