#!/usr/bin/env python
# coding: utf-8

# In[142]:


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


# In[104]:


data = pd.read_csv('solar_generation_data.csv', sep = ',')#, parse_dates=['date'])


# In[105]:


data.shape


# In[106]:


data.head()


# In[107]:


data.isna().sum()


# In[108]:


data = data.fillna(0)


# In[109]:


data.isna().sum()


# In[110]:


data.dtypes


# In[111]:


data.columns


# In[112]:


data.rename(columns={'Month ':'Month'}, inplace=True)


# In[113]:


data.Month


# In[114]:


data['Month'] = pd.to_datetime(data.Month, format='%b').dt.month


# In[115]:


data.tail(10)


# In[116]:


data["Temp Low"] = data["Temp Low"].replace('\u00b0','', regex=True)
data["Temp Hi"] = data["Temp Hi"].replace('\u00b0','', regex=True)


# In[117]:


data.tail(10)


# In[118]:


data["Temp Hi"] = pd.to_numeric(data["Temp Hi"], errors='coerce')
data["Temp Low"] = pd.to_numeric(data["Temp Low"], errors='coerce')


# In[119]:


data.dtypes


# In[120]:


data.head(10)


# In[121]:


#data['Temp_Avg_Far'] = data[['Temp Hi', 'Temp Low']].mean(axis=1)


# In[122]:


#data.head(10)


# In[123]:


#Convert Fahrenheit to Celsius
#def far_cel(temp_far):
#    temp_cel = (temp_far - 32) * 5 / 9
#    return temp_cel


# In[124]:


#data['Temp_Avg_Cel']=far_cel(data["Temp_Avg_Far"])


# In[125]:


#data.head(10)


# In[126]:


data = data.drop(columns =['Solar' ]) 


# In[127]:


data.tail(10)


# In[128]:


corr = data.corr()


# In[129]:


sns.set()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[130]:


data.describe()


# In[233]:


y= data['Power Generated in MW'].values
Xdata = data.drop(['Power Generated in MW', 'Rainfall in mm'], axis = 1)
X = Xdata.values
#X = StandardScaler().fit_transform(X)
model = RandomForestRegressor(n_jobs=-1)


# In[241]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[242]:


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


# In[243]:


labels = Xdata.columns
result = pd.DataFrame()
result['feature'] = labels
result['importance'] = model.feature_importances_
result.sort_values(by=['importance'], ascending=False, inplace=True)
result


# In[244]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('Training data', len(X_train), len(y_train))
print('Test data', len(X_test), len(y_test))


# In[245]:


dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X_train, y_train)


# In[246]:


Y_predicted = dt.predict(X_test)
dt.score(X_test, y_test)


# In[64]:


#r2_score(y_test, Y_predicted)


# In[255]:


filename = 'Solar_model.sav'
pickle.dump(dt, open(filename, 'wb'))


# In[247]:


pickle.dump(dt, open('solar_model.pkl','wb'))


# API Code

# In[248]:


# Solar
longitude = 142.110216
latitude = -19.461907

url = ('https://api.openweathermap.org/data/2.5/onecall?lat=-19.461907&lon=142.110216&units=imperial&appid=43e49f2fb4d17b806dfff389f21f4d27')
response = requests.get(url)

print(response.status_code)


# In[249]:


weather = response.json()
dailynorm = json_normalize(weather, 'daily')
#dailynorm.head(10)
df = pd.DataFrame(dailynorm)


# In[250]:


solar_df = df[['dt', 'temp.min', 'temp.max', 'clouds']].copy()
solar_df.head(10)


# In[251]:


solar_df['date'] = pd.to_datetime(solar_df['dt'],unit='s')
solar_df['day'] = solar_df['date'].dt.day
solar_df['month'] = solar_df['date'].dt.month
solar_df.head(10)


# In[252]:


solar_df.rename(columns={'temp.min':'Temp Low',
                          'temp.max':'Temp Hi',
                          'clouds':'Cloud Cover Percentage'}, 
                 inplace=True)
solar_df = solar_df.drop(['dt','date'], axis = 1)


# In[253]:


solar_df = solar_df.fillna(0)


# In[268]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[257]:


Xnew = solar_df.values


# In[258]:


p_pred = loaded_model.predict(Xnew)


# In[259]:


p_pred


# In[260]:


p_pred = pd.DataFrame(p_pred)


# In[261]:


p_pred.head()


# In[262]:


p_pred.dtypes


# In[207]:


#p_pred.rename(columns={'0':'Predicted Power'}, inplace=True)


# In[263]:


p_pred.columns = ['Predicted Power']


# In[264]:


p_pred.head()


# In[265]:


final_solar_df = pd.concat([solar_df, p_pred], axis = 1)


# In[266]:


print(final_solar_df)


# In[ ]:





# In[148]:

