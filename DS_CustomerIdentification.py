#Uptake Data Science Case Study
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns

#Import Data
Data_Train = pd.read_csv('train.csv')
Data_Test = pd.read_csv('test.csv')
Data_Marketing = pd.read_csv('zipCodeMarketingCosts.csv')
data_dict = open("dataDictionary.txt", "rb").read()
Data_Train.head()
col = Data_Train.columns 
print(col)

#Data Inspectipn and Analysis
#Remove data observations with bad address
#training:
Data_Train = Data_Train[Data_Train.mailcode != 'B']

#create a list to drop variables
ls_drop = ['mailcode','mdmaud','ageflag']
#zipcodes ending with - needs to be cleaned
Data_Train['zip']=Data_Train['zip'].str[:5]
Data_Test['zip']=Data_Test['zip'].str[:5]

#Replace spaces with Y as a place holder for encoding
#do this for all the variables in the first part of data
#We will use these variables for feature selection

Data_Train['has_chapter'] = Data_Train['has_chapter'].str.replace(' ','Y')
Data_Test['has_chapter'] = Data_Test['has_chapter'].str.replace(' ','Y')

Data_Train['recinhse'] = Data_Train['recinhse'].str.replace(' ','Y')
Data_Test['recinhse'] = Data_Test['recinhse'].str.replace(' ','Y')

Data_Train['recp3'] = Data_Train['recp3'].str.replace(' ','Y')
Data_Test['recp3'] = Data_Test['recp3'].str.replace(' ','Y')

Data_Train['recpgvg'] = Data_Train['recpgvg'].str.replace(' ','Y')
Data_Test['recpgvg'] = Data_Test['recpgvg'].str.replace(' ','Y')

Data_Train['recsweep'] = Data_Train['recsweep'].str.replace(' ','Y')
Data_Test['recsweep'] = Data_Test['recsweep'].str.replace(' ','Y')

Data_Train['solp3'] = Data_Train['solp3'].str.replace(' ','0')
Data_Test['solp3'] = Data_Test['solp3'].str.replace(' ','0')

Data_Train['solih'] = Data_Train['solih'].str.replace(' ','0')
Data_Test['solih'] = Data_Test['solih'].str.replace(' ','0')

#First pass to remove unwanted variables in the data
Data_Train = Data_Train.drop(ls_drop, axis=1)
Data_Test = Data_Test.drop(ls_drop, axis=1)

#Encoding the variables which are processed
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
Data_Train.has_chapter = labelencoder_X.fit_transform(Data_Train.has_chapter)
Data_Test.has_chapter = labelencoder_X.transform(Data_Test.has_chapter)

Data_Train.recinhse = labelencoder_X.fit_transform(Data_Train.recinhse)
Data_Test.recinhse = labelencoder_X.transform(Data_Test.recinhse)

Data_Train.recp3 = labelencoder_X.fit_transform(Data_Train.recp3)
Data_Test.recp3 = labelencoder_X.transform(Data_Test.recp3)

Data_Train.recpgvg = labelencoder_X.fit_transform(Data_Train.recpgvg)
Data_Test.recpgvg = labelencoder_X.transform(Data_Test.recpgvg)

Data_Train.recsweep = labelencoder_X.fit_transform(Data_Train.recsweep)
Data_Test.recsweep = labelencoder_X.transform(Data_Test.recsweep)

ls_drop = ['dob']
Data_Train = Data_Train.drop(ls_drop, axis=1)
Data_Test = Data_Test.drop(ls_drop, axis=1)

#Too many missing values for homeowner: may lead incorrect prediciton
#Dropping homeowner from data
ls_drop = ['homeownr']
Data_Train = Data_Train.drop(ls_drop, axis=1)
Data_Test = Data_Test.drop(ls_drop, axis=1)


#impute values for nan
list_impute = ['age','income_range', 'wealth1']
from sklearn.preprocessing import Imputer
imputer =  Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(Data_Train[list_impute])# upper bound is not included while specifying index columns
Data_Train[list_impute] = imputer.transform(Data_Train[list_impute])
#For testing Data
imputer =  Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(Data_Test[list_impute])# upper bound is not included while specifying index columns
Data_Test[list_impute] = imputer.transform(Data_Test[list_impute])

#Dropping childxx variables and keeping number of children and replacing nans with 0
ls_drop=['child03','child07','child12','child18']
Data_Train = Data_Train.drop(ls_drop, axis=1)
Data_Test = Data_Test.drop(ls_drop, axis=1)
Data_Train['numchld'] = Data_Train['numchld'].fillna(value=0)
Data_Test['numchld'] = Data_Test['numchld'].fillna(value=0)

#Replcae nans by zeros for other for customer response to other tye of mail order
ls_impute3 = ['mbcraft','mbgarden','mbbooks','mbcolect',
              'magfaml',	'magfem'	,'magmale',	'pubgardn',	'pubculin',	'pubhlth'	,'pubdoity',
              'pubnewfn'	,'pubphoto'	,'pubopp']
Data_Train[ls_impute3] = Data_Train[ls_impute3].fillna(value=0)
Data_Test[ls_impute3] = Data_Test[ls_impute3].fillna(value=0)

#Drop Data source: intuition and many missing values
ls_drop = ['datasrce']
Data_Train = Data_Train.drop(ls_drop, axis=1)
Data_Test = Data_Test.drop(ls_drop, axis=1)

#fill the rest of nans with 0
Data_Train = Data_Train.fillna(value=0)
Data_Test = Data_Test.fillna(value=0)


#Remove geocode
ls_drop = ['geocode']
Data_Train = Data_Train.drop(ls_drop, axis=1)
Data_Test = Data_Test.drop(ls_drop, axis=1)
#remove lifestyle source
ls_drop = ['lifesrc']
Data_Train = Data_Train.drop(ls_drop, axis=1)
Data_Test = Data_Test.drop(ls_drop, axis=1)



from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
Data_Train.source = labelencoder_X.fit_transform(Data_Train.source)
Data_Test.source = labelencoder_X.transform(Data_Test.source)

Data_Train.state = labelencoder_X.fit_transform(Data_Train.state)
Data_Test.state = labelencoder_X.transform(Data_Test.state)

Data_Train.domain = labelencoder_X.fit_transform(Data_Train.domain)
Data_Test.domain = labelencoder_X.transform(Data_Test.domain)

Data_Train.gender = labelencoder_X.fit_transform(Data_Train.gender)
Data_Test.gender = labelencoder_X.transform(Data_Test.gender)

Data_Train.solp3 = labelencoder_X.fit_transform(Data_Train.solp3)
Data_Test.solp3 = labelencoder_X.fit_transform(Data_Test.solp3)

Data_Train.solih = labelencoder_X.fit_transform(Data_Train.solih)
Data_Test.solih = labelencoder_X.fit_transform(Data_Test.solih)

Data_Train.major = labelencoder_X.fit_transform(Data_Train.major)
Data_Test.major = labelencoder_X.transform(Data_Test.major)

Data_Train.collect1 = labelencoder_X.fit_transform(Data_Train.collect1)
Data_Test.collect1 = labelencoder_X.transform(Data_Test.collect1)

Data_Train.veterans = labelencoder_X.fit_transform(Data_Train.veterans)
Data_Test.veterans = labelencoder_X.transform(Data_Test.veterans)

Data_Train.bible = labelencoder_X.fit_transform(Data_Train.bible)
Data_Test.bible = labelencoder_X.transform(Data_Test.bible)

Data_Train.catlg = labelencoder_X.fit_transform(Data_Train.catlg)
Data_Test.catlg = labelencoder_X.transform(Data_Test.catlg)

Data_Train.homee = labelencoder_X.fit_transform(Data_Train.homee)
Data_Test.homee = labelencoder_X.transform(Data_Test.homee)

Data_Train.stereo = labelencoder_X.fit_transform(Data_Train.stereo)
Data_Test.stereo = labelencoder_X.transform(Data_Test.stereo)

Data_Train.cdplay = labelencoder_X.fit_transform(Data_Train.cdplay)
Data_Test.cdplay = labelencoder_X.transform(Data_Test.cdplay)

Data_Train.pcowners = labelencoder_X.fit_transform(Data_Train.pcowners)
Data_Test.pcowners = labelencoder_X.transform(Data_Test.pcowners)

Data_Train.photo = labelencoder_X.fit_transform(Data_Train.photo)
Data_Test.photo = labelencoder_X.transform(Data_Test.photo)

Data_Train.crafts = labelencoder_X.fit_transform(Data_Train.crafts)
Data_Test.crafts = labelencoder_X.transform(Data_Test.crafts)

Data_Train.fisher = labelencoder_X.fit_transform(Data_Train.fisher)
Data_Test.fisher = labelencoder_X.transform(Data_Test.fisher)

Data_Train.gardenin = labelencoder_X.fit_transform(Data_Train.gardenin)
Data_Test.gardenin = labelencoder_X.transform(Data_Test.gardenin)

Data_Train.boats = labelencoder_X.fit_transform(Data_Train.boats)
Data_Test.boats = labelencoder_X.transform(Data_Test.boats)

Data_Train.walker = labelencoder_X.fit_transform(Data_Train.walker)
Data_Test.walker = labelencoder_X.transform(Data_Test.walker)

Data_Train.kidstuff = labelencoder_X.fit_transform(Data_Train.kidstuff)
Data_Test.kidstuff = labelencoder_X.transform(Data_Test.kidstuff)

Data_Train.cards = labelencoder_X.fit_transform(Data_Train.cards)
Data_Test.cards = labelencoder_X.transform(Data_Test.cards)

Data_Train.plates = labelencoder_X.fit_transform(Data_Train.plates)
Data_Test.plates = labelencoder_X.transform(Data_Test.plates)

Data_Train.pepstrfl = labelencoder_X.fit_transform(Data_Train.pepstrfl)
Data_Test.pepstrfl = labelencoder_X.transform(Data_Test.pepstrfl)

#drop adates
list_adates = ['adate_2',	'adate_3','adate_4',	'adate_5','adate_6','adate_7',
              'adate_8','adate_9','adate_10','adate_11','adate_12','adate_13',
              'adate_14','adate_15','adate_16','adate_17','adate_18','adate_19',
              'adate_20','adate_21','adate_22','adate_23','adate_24']

Data_Train = Data_Train.drop(list_adates, axis=1)
Data_Test = Data_Test.drop(list_adates, axis=1)

#Drop rdates
list_rdates = ['rdate_3','rdate_4',	'rdate_5','rdate_6','rdate_7',
              'rdate_8','rdate_9','rdate_10','rdate_11','rdate_12','rdate_13',
              'rdate_14','rdate_15','rdate_16','rdate_17','rdate_18','rdate_19',
              'rdate_20','rdate_21','rdate_22','rdate_23','rdate_24']

Data_Train = Data_Train.drop(list_rdates, axis=1)
Data_Test = Data_Test.drop(list_rdates, axis=1)

#remove other dates as well since the factors used in coding are derived from dates
list_dates=['date','minrdate','maxrdate','lastdate', 'fistdate','lastdate','nextdate']
Data_Train = Data_Train.drop(list_dates, axis=1)
Data_Test = Data_Test.drop(list_dates, axis=1)

#majority of mdmauds are X . they will not have significant influence on the prediction
#removing them too
list_mdmaud = ['mdmaud_r','mdmaud_f','mdmaud_a']
Data_Train = Data_Train.drop(list_mdmaud, axis=1)
Data_Test = Data_Test.drop(list_mdmaud, axis=1)

#removing rfa status and encoding the rfa status codes for 97nk
list_rfa=['rfa_2','rfa_3','rfa_4',	'rfa_5','rfa_6','rfa_7',
              'rfa_8','rfa_9','rfa_10','rfa_11','rfa_12','rfa_13',
              'rfa_14','rfa_15','rfa_16','rfa_17','rfa_18','rfa_19',
              'rfa_20','rfa_21','rfa_22','rfa_23','rfa_24']
Data_Train = Data_Train.drop(list_rfa, axis=1)
Data_Test = Data_Test.drop(list_rfa, axis=1)

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
Data_Train.rfa_2r = labelencoder_X.fit_transform(Data_Train.rfa_2r)
Data_Test.rfa_2r = labelencoder_X.transform(Data_Test.rfa_2r)

Data_Train.rfa_2f = labelencoder_X.fit_transform(Data_Train.rfa_2f)
Data_Test.rfa_2f = labelencoder_X.transform(Data_Test.rfa_2f)

Data_Train.rfa_2a = labelencoder_X.fit_transform(Data_Train.rfa_2a)
Data_Test.rfa_2a = labelencoder_X.transform(Data_Test.rfa_2a)

#enciding geocode2 was cusing an error: dropping for now since short ontime
drop_geocode = ['geocode2']
Data_Train = Data_Train.drop(drop_geocode, axis=1)
Data_Test = Data_Test.drop(drop_geocode, axis=1)

#check data:
Data_Train.shape
Data_Train.dtypes




#At this point I get an error which I do not have enough time to debug
#I would have done the following steps. 


#Step 2: Builing prediction models
#Random forest classification - without feature reduction
y_train = Data_Train.responded
y_test = Data_Test.market

list_dropResp = ['responded','zip', 'noexch', 'cluster', 'pets','amount']
X_train = Data_Train.drop(list_dropResp , axis=1)
list_dropResp = ['market','zip', 'noexch', 'cluster', 'pets']
X_test = Data_Test.drop(list_dropResp , axis=1)
g = X_train.columns.to_series().groupby(X_train.dtypes).groups
y_amount = Data_Train['amount']
import csv
#export the prediction
pd.DataFrame(X_train).to_csv('X_train.csv', index=False, header=True)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False, header=True)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False, header=True)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False, header=True)
pd.DataFrame(y_amount).to_csv('y_amount.csv', index=False, header=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score

classifier = RandomForestClassifier()
classifier = classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
ac = accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test, classifier.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d")

#univariate feature selection using SelectKBest 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#find best scored 100 features
select_feature = SelectKBest(chi2,k=100).fit(X_train, y_train)
feature_List = X_train.columns
print('Score list: ', select_feature.scores_)
print('Feature list: ', X_train.columns)

X_train_2 = select_feature.transform(X_train)
X_test_2 = select_feature.transform(X_test)
#Random forest
classifier_2 = RandomForestClassifier()
classifier_2 = classifier_2.fit(X_train_2, y_train)
ac_2 = accuracy_score(y_test, classifier_2.predict(X_test_2))
print('Accuracy is: ', ac_2)
cm_2 = confusion_matrix(y_test, classifier_2.predict(X_test_2))
sns.heatmap(cm_2, annot=True,fmt="d")
y_predict_selectKBest = classifier_2.predict(X_test_2)

#Recursive feature elimination with random forest
from sklearn.feature_selection import RFE
classifier_RFE = RandomForestClassifier()
rfe = RFE(estimator=classifier_RFE, n_features_to_select = 100, step=1)
rfe = rfe.fit(X_train,y_train)
print('Chosen best 100 feature by RFE: ',X_train.columns[rfe.support_])
RFE_Features = X_train.columns[rfe.support_]
X_test_Rfe = X_test[RFE_Features]
X_train_Rfe = X_train[RFE_Features]
rfe_newfit =classifier_RFE.fit(X_train_Rfe, y_train)
rfe_predict = rfe_newfit.predict(X_test_Rfe)
acc_RFE = accuracy_score(y_test, rfe_predict)
print('Accuracy is: ', acc_RFE)
cm_RFE = confusion_matrix(y_test, rfe_predict)
sns.heatmap(cm_RFE, annot=True,fmt="d")
#Data Visualization
corr_rfe = X_train_Rfe.corr()
sns.heatmap(corr_rfe,
            xticklabels=corr_rfe.columns.values,
            yticklabels=corr_rfe.columns.values)


#Recursive feature elimination with cross validation
#helps select optimal number of features usinf RF algorithm
from sklearn.feature_selection import RFECV
classifier_RFECV = RandomForestClassifier()
rfecv = RFECV(estimator=classifier_RFECV, step=1, cv=5, scoring='accuracy')
rfecv = rfecv.fit(X_train,y_train)
print('Optimal number of features: ',rfecv.n_features_)
print('Best features: ',X_train.columns[rfecv.support_])
RFECV_Features = X_train.columns[rfecv.support_]
X_test_RFECV = X_test[RFECV_Features]
X_train_RFECV = X_train[RFECV_Features]
rfecv_newfit =classifier_RFECV.fit(X_train_RFECV, y_train)
rfecv_predict = rfecv_newfit.predict(X_test_RFECV)
acc_RFECV = accuracy_score(y_test, rfecv_predict)
print('Accuracy is: ', acc_RFECV)
#check accuracy with plot
plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('CV score')
plt.plot(range(1,len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.show()



#Using SVM
from sklearn.svm import SVC
import matplotlib.cm as cm

#from adspy_shared_utilities import plot_class_regions_for_classifier
#unmormalized

classifier_SVC = SVC()
classifier_SVC = classifier_SVC.fit(X_train, y_train)

ac_SVC = accuracy_score(y_test, classifier_SVC.predict(X_test))
print('Accuracy is: ', ac_SVC)
cm_SVC = confusion_matrix(y_test, classifier_SVC.predict(X_test))
sns.heatmap(cm_SVC, annot=True,fmt="d")

#normalized
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
classifier_SVC = SVC()
classifier_SVC = classifier_SVC.fit(X_train_std, y_train)

ac_SVC = accuracy_score(y_test, classifier_SVC.predict(X_test_std))
print('Accuracy is: ', ac_SVC)
cm_SVC = confusion_matrix(y_test, classifier_SVC.predict(X_test_std))
sns.heatmap(cm_SVC, annot=True,fmt="d")

#Parameter tuning gamma and penalty
for this_gamma in [0.01,0.1,0.5,1,5,50]:
    for this_C in [0.1,1,5,15,100]:
        classifier_SVC = SVC(kernel='rbf', gamma=this_gamma,C = this_C)
        classifier_SVC = classifier_SVC.fit(X_train_std, y_train)
        acc_SVC_Train = accuracy_score(y_train, classifier_SVC.predict(X_train_std))
        acc_SVC = accuracy_score(y_test, classifier_SVC.predict(X_test_std))
        print('gamma %f C %f Train accuracy %f test accuracy %f' 
              %(this_gamma,this_C,acc_SVC_Train,acc_SVC))

#Model 2: Linear model for amount
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
regression = linear_model.LinearRegression()
regression = regression.fit(X_train, y_amount)
y_pred_amount = regression.predict(X_test)
pd.DataFrame(y_pred_amount).to_csv('y_pred_amount.csv', index=False, header=True)
#Write data to files
import csv
#export the prediction
pd.DataFrame(y_predict).to_csv('y_predict.csv', index=False, header=True)
#export selectKBest prediction
pd.DataFrame(y_predict_selectKBest).to_csv('y_predict_KBest.csv', index=False, header=True)
#Export prediction under RFE
pd.DataFrame(rfe_predict).to_csv('y_predict_RFE.csv', index=False, header=True)
#Export prediction under RFECV
pd.DataFrame(rfecv_predict).to_csv('y_predict_RFECV.csv', index=False, header=True)
#export the feature selection file
featureSel = "C:/Users/user/Downloads/FeatureSel.csv"
with open(featureSel, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in feature_List:
        writer.writerow([val])

#RFE feature file
featureSel_RFE = "C:/Users/user/Downloads/FeatureSel_RFE.csv"
with open(featureSel_RFE, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in RFE_Features:
        writer.writerow([val])
        
#export the zip  column of test data: cleaned
zip_cleaned = Data_Test['zip']    
pd.DataFrame(zip_cleaned).to_csv('zip_cleaned.csv', index=False, header=True)  
  
'''

#After having a prediction on the responses, 
#I would have used linear program to identy the best sponsors whihc maximizes the profit
#The prediction model would have been my constraint in the linear programming model 