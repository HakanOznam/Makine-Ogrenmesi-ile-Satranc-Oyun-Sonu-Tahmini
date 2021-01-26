
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_excel("games.xlsx")

#hamle sayısı <10 olan oyunları veritabanından silme işlemi
data=data.drop(data.loc[data['turns']<=10].index)
data=data.reset_index(drop=True)

#eksik veri kontrolü
data.isnull().sum()

#IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
testmissing = data.iloc[:9000,2:4]
trainmissing = data.iloc[9000:,2:4]
imp = IterativeImputer(random_state=0)
imp.fit(trainmissing)
data.iloc[:9000,2:4] = np.round(imp.transform(testmissing))

#Label Encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#rated
data.iloc[:,1] = le.fit_transform(data.iloc[:,1])

#victory status
ohe = preprocessing.OneHotEncoder()
vicstatus = data.iloc[:,4].to_numpy()
vicstatus = vicstatus.reshape(-1,1)
ohe_vicstatus = ohe.fit_transform(vicstatus).toarray()
dfvicstatus=pd.DataFrame(data=ohe_vicstatus,columns=["draw","mate","outoftime","resign"])

#opening_eco
eco = data['opening_eco']
for i in range(len(eco)):  
    eco[i]=eco[i].replace("A","1")
    eco[i]=eco[i].replace("B","2")
    eco[i]=eco[i].replace("C","3")
    eco[i]=eco[i].replace("D","4")
    eco[i]=eco[i].replace("E","5")

eco=pd.to_numeric(eco)

"""Makine öğrenmesi uygulamak üzere veri kümesinin öznitelik,etiket olarak bölünmesi"""

X = pd.concat([data["rated"],data["game_time"],data["turns"],dfvicstatus,data["black_rating"],data["white_rating"]],axis=1)
y = data["winner"]
y = le.fit_transform(y)
classes =["black","draw","white"]


"""#Visualization"""

#game_time
plt.hist(X["game_time"],bins=500,color='g',label="game time")
plt.xlabel("time")
plt.ylabel("distribution")
plt.legend()
plt.title("Game Time")
plt.show()

#turns
plt.hist(X["turns"],bins=100,color='r',label="turns")
plt.xlabel("number of moves")
plt.ylabel("distribution")
plt.legend()
plt.title("turns")
plt.show()

#white-black rating
plt.hist([X["white_rating"],X["black_rating"]],color=['#0c457d','#0ea7b5'],label=["white","black"])
plt.xlabel("rating")
plt.ylabel("distribution")
plt.legend()
plt.title("white-black rating")
plt.show()

#opening_eco
df_eco = pd.DataFrame(eco)

plt.scatter(range(1,3747),df_eco[(df_eco["opening_eco"] >=100) & (df_eco["opening_eco"] < 200)].values,color="red",label="A00-A99")
plt.scatter(range(1,5048),df_eco[(df_eco["opening_eco"] >=200) & (df_eco["opening_eco"] < 300)].values,color="blue",label="B00-B99")
plt.scatter(range(1,7449),df_eco[(df_eco["opening_eco"] >=300) & (df_eco["opening_eco"] < 400)].values,color="green",label="C00-C99")
plt.scatter(range(1,2633),df_eco[(df_eco["opening_eco"] >=400) & (df_eco["opening_eco"] < 500)].values,color="purple",label="D00-D99")
plt.scatter(range(1,505),df_eco[(df_eco["opening_eco"] >=500) & (df_eco["opening_eco"] < 600)].values,color="gray",label="E00-E99")
plt.xlabel("distribution")
plt.ylabel("Opening eco codes")
plt.legend()
plt.show()


""" Verilerin ölçeklendirilmesi """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sc = sc.fit_transform(X)
#KFold uygulaması
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay ,classification_report

""" Makine öğrenmesi modellerinin sonuçlarını yazdırma fonksiyonu"""
def output(est,X,color):
   
    for model in est:
        accu = cross_val_score(estimator = model, X = X, y=y,cv=5)
        y_pred = cross_val_predict(model,X,y,cv=5)
        conf_matrix = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=["black","draw","white"])
        disp.plot(cmap=color)
        disp.ax_.set_title(model)
        print("-------- ",model," report -------- \n")
        print(classification_report(y,y_pred,target_names=classes))
        print("\n")

""""Modellerin import edilmesi"""
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


"""Random Forest Grid_SearchCv """
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

parameters = [{'n_estimators': [50,100,150,200], 'criterion':['gini',"entropy"]}]
grids = GridSearchCV(estimator = rfc, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grids.fit(X_sc, y)
best_accuracy , best_parameter = grids.best_score_ , grids.best_params_
print('En iyi acc: ', best_accuracy)
print('En iyi acc veren parametreler: ', best_parameter,"\n")
grid_rfc = RandomForestClassifier(n_estimators=150,criterion="entropy")


"""KNN Grid_SearchCv """
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

parameters = [{'n_neighbors': [3,5,7,9,11,13,15,17,19,21], 'weights':['uniform',"distance"],"metric" : ["euclidean","manhattan","minkowski"]}]
grids = GridSearchCV(estimator = knn, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grids.fit(X_sc, y)
best_accuracy , best_parameter = grids.best_score_ , grids.best_params_
print('En iyi acc: ', best_accuracy)
print('En iyi acc veren parametreler: ', best_parameter,"\n")
grid_knn = KNeighborsClassifier(n_neighbors=21,weights="distance",metric="manhattan")


"""SVM Grid_SearchCv """
from sklearn.svm import SVC
svm = SVC()

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel':['linear']}, {'C': [0.25, 0.5, 0.75, 1], 'kernel':['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]
grids = GridSearchCV(estimator = svm, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grids.fit(X_sc, y)
best_accuracy , best_parameter = grids.best_score_ , grids.best_params_
print('En iyi acc: ', best_accuracy)
print('En iyi acc veren parametreler: ', best_parameter,"\n")
grid_svm = SVC(C=0.75,kernel="linear")


"""Modellerin sonuçlarının yazdırılması"""

"""Parametre kullanılmadan modellerin uygulanması""" 
est = [logreg,clf,rfc,knn,svm,gnb]
output(est,X_sc,"pink")

"""Grid_searchCV uygulanan üç modelin sonuçları """
grid_est = [grid_rfc,grid_knn,grid_svm]
output(grid_est,X_sc,"hot")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_lda = lda.fit_transform(X_sc, y)

"""LDA ve grid_searchCV uygulanan modellerin sonuçları"""
grid_lda_est = [logreg,clf,grid_rfc,grid_knn,grid_svm,gnb]
output(grid_lda_est,X_lda,"bone")

"""Yeni veri kümesiyle modellerin uygulanması"""
newX = pd.concat([data["turns"],dfvicstatus],axis=1)
newX_sc = sc.fit_transform(newX)
lda = LDA(n_components = 2)
newX_lda = lda.fit_transform(newX_sc, y)

newdata_est = [logreg,clf,grid_rfc,grid_knn,grid_svm,gnb]
output(newdata_est,newX_lda,"cubehelix")

