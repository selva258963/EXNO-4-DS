# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
 STEP 1:Read the given Data.
 
 STEP 2:Clean the Data Set using Data Cleaning Process.
 
 STEP 3:Apply Feature Scaling for the feature in the data set.
 
 STEP 4:Apply Feature Selection for the feature in the data set.
 
 STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
NAME : SELVAMUTHU KUMARAN V
REFERENCE NO : 212222040151
```
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/ca77f2ff-56d2-497a-b7c8-4758535ecd0d)
```
data.isnull().sum()
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/df566d53-be78-4d89-bd82-d6951845258b)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/6fd563b8-3d3f-422f-ac06-36025c5aeb33)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/3f1b25f6-d6cc-4cf7-8296-69f41a06529e)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/13028cf0-99aa-4049-83be-f9060f5b9bbf)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/23eec7bf-f38e-41db-a870-3d8df8103ea2)
```
data2
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/521378b5-9719-4fbd-9457-fef525c9243a)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/7dc57367-3e63-47e6-97cc-b793d9c73813)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/47f4b895-72ce-4d25-bdc2-ae76a484fe26)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/57ea8147-82b1-41d0-9e43-756165949ded)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/c03713b9-12f8-41b1-bd30-21e93baef57c)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/4c44ca14-3c12-4e8e-8455-f3a89fa9b2da)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/f44b8500-30a4-4780-8594-198ad38248be)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/7a804881-6539-434f-8707-6af6b812bafc)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/a01da385-bca7-47b7-a4c1-2e14f779c859)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/6a6242d6-e97e-4214-a850-905342ec9eb7)

```
data.shape
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/b3d24141-7608-49e1-ab3a-ed195a1ae7fe)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/b89ccdd7-12db-4edd-b017-b5eeeb04e033)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/9fe99701-4887-490c-b0ae-7e913681954b)
```
tips.time.unique()
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/62a26d0b-1d1f-462c-898f-647229d3d613)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/c4f24078-6573-44c9-8ff9-ef7197486de8)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/22008650/EXNO-4-DS/assets/122548204/2d91381c-c939-4980-ab3e-ae4aaf828aab)


# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
