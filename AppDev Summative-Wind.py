#!/usr/bin/env python
# coding: utf-8

# In[353]:

# libraries
import pandas as pd
import numpy as np
import requests
import json
from pandas.io.json import json_normalize

#import matplotlib
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

import pickle



# In[312]:


data = pd.read_csv('wind_generation_data.csv', sep = ',')#, parse_dates=['date'])


# In[313]:


data.shape


# In[314]:


data ['days'] = data.index


# In[315]:


data.head(10)


# In[316]:


#count = 0
def plus(numb):
    count = numb + 1
    return count


# In[317]:


data.dtypes


# In[318]:


data['dayg'] = plus(data['days'])


# In[319]:


def month(x):
    y = 31
    p, q = divmod(x, y)
    return (p)


# In[320]:


data['months'] = month(data['days'])


# In[321]:


data['month'] = plus(data['months'])


# In[322]:


data.head(40)


# In[323]:


def dayum(x):
    #while x != 0:
    y = 31
    p, q = divmod(x, y)
    #print (q)
    g = q ++ 1
    return (g)


# In[324]:


data['day'] = dayum(data['days'])


# In[325]:


data.head(40)


# In[326]:


data = data.drop(['months', 'days', 'dayg'], axis = 1)


# In[327]:


data.head(40)


# In[328]:


data.isna().sum()


# In[329]:


data.dtypes


# In[330]:


data.columns


# In[331]:


corr = data.corr()


# In[333]:


sns.set()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[334]:


data.describe()


# In[335]:


y= data['Power Output'].values
Xdata = data.drop(['Power Output'], axis = 1)
X = Xdata.values
#X = StandardScaler().fit_transform(X)
model = RandomForestRegressor(n_jobs=-1)


# In[336]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[337]:


estimators = np.arange(1, 12, 2)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.title("Regressor Plot")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[338]:


labels = Xdata.columns
result = pd.DataFrame()
result['feature'] = labels
result['importance'] = model.feature_importances_
result.sort_values(by=['importance'], ascending=False, inplace=True)
result


# In[339]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('Training data', len(X_train), len(y_train))
print('Test data', len(X_test), len(y_test))


# In[348]:


dt = DecisionTreeRegressor(max_depth=4)
dt.fit(X_train, y_train)


# In[349]:


Y_predicted = dt.predict(X_test)
dt.score(X_test, y_test)


# In[113]:


#filename = 'Wind_model.sav'
#pickle.dump(dt, open(filename, 'wb'))


# In[351]:


pickle.dump(dt, open('wind_model.pkl','wb'))


# API Code

# In[354]:


# Wind
longitude = 53.556563
latitude = 8.598084

url = ('https://api.openweathermap.org/data/2.5/onecall?lat=8.598084&lon=53.556563&units=imperial&appid=43e49f2fb4d17b806dfff389f21f4d27')
response = requests.get(url)

print(response.status_code)


# In[355]:


weather = response.json()
dailynorm = json_normalize(weather, 'daily')
df = pd.DataFrame(dailynorm)


# In[356]:


wind_df = df[['dt', 'wind_speed', 'wind_deg']].copy()
wind_df.head(10)


# In[359]:


wind_df['date'] = pd.to_datetime(wind_df['dt'],unit='s')
wind_df['day'] = wind_df['date'].dt.day
wind_df['month'] = wind_df['date'].dt.month
wind_df.head(10)


# In[361]:


wind_df.rename(columns={'wind_speed':'wind speed', 'wind_deg':'direction'}, inplace=True)
wind_df = wind_df.drop(['dt','date'], axis = 1)


# In[362]:


wind_df = wind_df.fillna(0)


# In[363]:


model = pickle.load(open('wind_model.pkl','rb'))


# In[364]:


Xnew = wind_df.values


# In[365]:


p_pred = model.predict(Xnew)


# In[366]:


p_pred


# In[367]:


p_pred = pd.DataFrame(p_pred)


# In[368]:


p_pred.columns = ['Predicted Power']


# In[369]:


final_wind_df = pd.concat([wind_df, p_pred], axis = 1)


# In[371]:


print(final_wind_df)


# In[ ]:




