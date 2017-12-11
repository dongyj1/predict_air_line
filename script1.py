import numpy as np
import pandas as pd 
import csv 
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

print('start running...')
u_cols = ['id','airline_name', 'author_country', 'cabin_flown', 'date', 'aircraft', 'type_traveller', 'route', 'overall_rating', 'seat_comfort_rating', 'cabin_staff_rating', 
'food_beverages_rating','inflight_entertainment_rating','ground_service_rating', 'wifi_connectivity_rating', 'value_money_rating', 'recommended']

cate_col = ['airline_name', 'author_country',  'cabin_flown', 'date', 'aircraft','type_traveller', 'route']

rating_cols = ['overall_rating', 'seat_comfort_rating', 'cabin_staff_rating', 
'food_beverages_rating','inflight_entertainment_rating','ground_service_rating', 'wifi_connectivity_rating', 'value_money_rating']

df_train = pd.read_csv('train.csv', sep=',' , quoting=csv.QUOTE_ALL, usecols=u_cols, dtype=str).fillna('0')
df_test = pd.read_csv('test.csv', sep=',' , quoting=csv.QUOTE_ALL, usecols=u_cols, dtype=str).fillna('0')

categories = pd.concat([df_train.loc[:,cate_col], df_test.loc[:,cate_col]])

print(df_train.shape)
print(df_test.shape)
print(categories.shape)
# categorical_features = range(3)
# les = []
# cate_names = {}


for cate in cate_col:

    le = preprocessing.LabelEncoder()
    categories[cate] = le.fit_transform(categories[cate])
    df_train[cate] = le.transform(df_train[cate])
    df_test[cate] = le.transform(df_test[cate])
    # cate_names[cate] = le.classes_
    # les.append(le)


encoder = preprocessing.OneHotEncoder()
encoder.fit(categories.loc[:,cate_col])
categories_train = encoder.transform(df_train.loc[:,cate_col])
categories_test = encoder.transform(df_test.loc[:,cate_col])

print(categories_train.shape)
print(categories_test.shape)

rating_train = df_train.loc[:,rating_cols].apply(pd.to_numeric)
rating_train[np.isnan(rating_train)] = 0

rating_test = df_test.loc[:,rating_cols].apply(pd.to_numeric)
rating_test[np.isnan(rating_test)] = 0
print(type(categories_train))

X_train = pd.concat( [pd.DataFrame(categories_train.toarray()), rating_train], axis=1)
X_test = pd.concat( [pd.DataFrame(categories_test.toarray()), rating_test], axis=1 )
y_train = df_train['recommended'].apply(pd.to_numeric)

y_test = pd.read_csv('result_linear_regression.csv', sep=',' , quoting=csv.QUOTE_ALL, usecols=['recommended'], dtype=str).apply(pd.to_numeric)
ids = df_test['id'].values
sample_leaf_options = [1,5,10,50,100,200,500]
# X_train = scale(X_train)
# X_test = scale(X_test)
# logistic = LogisticRegression()
# logistic.fit(X_train,y_train)
# y_predict = logistic.predict(X_test)

# for leaf_size in sample_leaf_options:
#     rf = RandomForestClassifier(n_estimators = 500, oob_score = True, max_features = "auto",min_samples_leaf = leaf_size)
#     rf.fit(X_train, y_train)
#     print("AUC - ROC : ", rf.score(X_test, y_test), leaf_size)
parameters = {'n_estimators':(1000),'max_features':(None, 9, 6, 'auto'),'max_depth':(None, 24, 16),'min_samples_split': (2, 4, 8),'min_samples_leaf': (16, 4, 12)}

rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, cv=10, n_jobs=2)
clf.fit(X_train, y_train)
# y_predict = rf.predict(X_test)
X_1, X_2, y_1, y_2 = train_test_split(X_train, y_train, test_size=0.2)
print(clf.best_score_, clf.score(X_2, y_2), clf.best_params_)
y_predict = clf.predict(X_test)
# compare results
y_test = pd.read_csv('result_linear_regression.csv', sep=',' , quoting=csv.QUOTE_ALL, usecols=['recommended'], dtype=str).apply(pd.to_numeric)


print(metrics.accuracy_score(y_test, y_predict))

ids = df_test['id'].values

df_predict = pd.DataFrame({'id':ids.tolist(),'recommended':y_predict})
df_predict.to_csv('result.csv', sep=',', index=False)

