#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# # Train Data

# In[2]:


df_train = pd.read_excel('Data_Train.xlsx')
df_train.head()


# In[3]:


df_train.info()


# In[4]:


df_train['Duration'].value_counts()


# In[5]:


df_train.isnull().sum()


# In[6]:


df_train = df_train.dropna()


# # EDA

# In[7]:


df_train['Journey_Day'] = pd.to_datetime(df_train['Date_of_Journey'], format="%d/%m/%Y").dt.day


# In[8]:


df_train['Journey_Month'] = pd.to_datetime(df_train['Date_of_Journey'], format="%d/%m/%Y").dt.month


# In[9]:


df_train.head()


# In[10]:


df_train.drop(['Date_of_Journey'], axis=1, inplace=True)


# In[11]:


df_train['Departure_hours'] = pd.to_datetime(df_train['Dep_Time']).dt.hour


# In[12]:


df_train['Departure_min'] = pd.to_datetime(df_train['Dep_Time']).dt.minute


# In[13]:


df_train.drop(['Dep_Time'], axis=1, inplace=True)


# In[14]:


df_train['Arrival_hours'] = pd.to_datetime(df_train['Arrival_Time']).dt.hour
df_train['Arrival_Minutes'] = pd.to_datetime(df_train['Arrival_Time']).dt.minute
df_train.drop(['Arrival_Time'],axis=1, inplace=True)
df_train.head()


# In[15]:


duration = list(df_train['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]
            
duration_minutes = []
duration_hours = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = 'h')[0]))
    duration_minutes.append(int(duration[i].split(sep = 'm')[0].split()[-1]))


# In[16]:


df_train['Duration_hours'] = duration_hours
df_train['Duration_miutes'] = duration_minutes
df_train.drop(['Duration'], axis=1, inplace=True)


# In[17]:


df_train.head()


# # Handling Categorical Features

# In[18]:


df_train['Airline'].value_counts()


# In[19]:


sns.catplot(y = "Price", x = "Airline", data = df_train.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3)


# In[20]:


Airlines = pd.get_dummies(df_train['Airline'], drop_first=True)
Airlines.head()


# In[21]:


df_train['Source'].value_counts()


# In[22]:


sns.catplot(y = "Price", x = "Source", data= df_train.sort_values("Price", ascending=False), kind = "boxen", height=6, aspect=3)
plt.show()


# In[23]:


Sources = pd.get_dummies(df_train['Source'], drop_first=True)
Sources.head()


# In[24]:


df_train['Destination'].value_counts()
#for i in range(len(df_train)):
#    if df_train['Destination'][i] == "Delhi":
#        df_train['Destination'][i] = 'New Delhi'


# In[25]:


sns.catplot(y = "Price", x = "Destination", data= df_train.sort_values("Price", ascending=False), kind = "boxen", height=6, aspect=3)
plt.show()


# In[26]:


Destinations = pd.get_dummies(df_train['Destination'], drop_first=True)
Destinations.head()


# In[27]:


df_train['Route']


# In[28]:


df_train.drop(['Route', 'Additional_Info'], axis=1, inplace=True)


# In[29]:


df_train['Total_Stops'].value_counts()


# In[30]:


df_train.replace({'non-stop': 0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}, inplace=True)


# In[31]:


df_train.head()


# In[32]:


df_train = pd.concat([df_train, Airlines, Sources, Destinations], axis=1)
df_train.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)
pd.set_option('display.max_columns', None)
df_train.head()


# # Test Data

# In[33]:


df_test = pd.read_excel('Test_set.xlsx')
df_test.head()


# In[34]:


df_test.isnull().sum()


# In[35]:


df_test["Journey_day"] = pd.to_datetime(df_test.Date_of_Journey, format="%d/%m/%Y").dt.day
df_test["Journey_month"] = pd.to_datetime(df_test["Date_of_Journey"], format = "%d/%m/%Y").dt.month
df_test.drop(["Date_of_Journey"], axis = 1, inplace = True)
df_test.head()


# In[36]:


df_test["Departure_hours"] = pd.to_datetime(df_test["Dep_Time"]).dt.hour
df_test["Departure_min"] = pd.to_datetime(df_test["Dep_Time"]).dt.minute
df_test.drop(["Dep_Time"], axis = 1, inplace = True)
df_test.head()


# In[37]:


df_test["Arrival_hours"] = pd.to_datetime(df_test.Arrival_Time).dt.hour
df_test["Arrival_Minutes"] = pd.to_datetime(df_test.Arrival_Time).dt.minute
df_test.drop(["Arrival_Time"], axis = 1, inplace = True)
df_test.head()


# In[38]:


duration = list(df_test['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]
            
duration_minutes = []
duration_hours = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = 'h')[0]))
    duration_minutes.append(int(duration[i].split(sep = 'm')[0].split()[-1]))


# In[39]:


df_test['Duration_hours'] = duration_hours
df_test['Duration_miutes'] = duration_minutes
df_test.drop(['Duration'], axis=1, inplace=True)
df_test.head()


# In[40]:


Airlines = pd.get_dummies(df_test['Airline'], drop_first=True)
Sources = pd.get_dummies(df_test['Source'], drop_first=True)
Destinations = pd.get_dummies(df_test['Destination'], drop_first=True)

df_test.drop(['Route', 'Additional_Info'], axis=1, inplace=True)
df_test.replace({'non-stop': 0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}, inplace=True)
df_test = pd.concat([df_test, Airlines, Sources, Destinations], axis=1)
df_test.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)
pd.set_option('display.max_columns', None)
df_test.head()


# # Feature Selection

# In[41]:


y = df_train['Price']
X = df_train.drop(['Price'], axis=1)


# In[42]:


plt.figure(figsize = (25, 25))
sns.heatmap(df_train.corr(), annot=True)


# In[43]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[44]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# # Model Training

# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[46]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)


# In[47]:


y_pred = reg_rf.predict(X_test)


# In[48]:


reg_rf.score(X_train, y_train)


# In[49]:


reg_rf.score(X_test, y_test)


# In[50]:


sns.distplot(y_test-y_pred)
plt.show()


# In[51]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[52]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[53]:


metrics.r2_score(y_test, y_pred)


# # Hyperparameter Tuning

# In[54]:


from sklearn.model_selection import RandomizedSearchCV


# In[55]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[56]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[57]:


rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[58]:


rf_random.fit(X_train,y_train)


# In[59]:


rf_random.best_params_


# In[60]:


prediction = rf_random.predict(X_test)


# In[61]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[62]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[63]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# # Saving The Model

# In[64]:


import pickle
file = open('flight_rf.pkl', 'wb')
pickle.dump(reg_rf, file)


# In[66]:


model = open('flight_rf.pkl','rb')
forest = pickle.load(model)

y_prediction = forest.predict(X_test)
metrics.r2_score(y_test, y_prediction)


# In[ ]:




