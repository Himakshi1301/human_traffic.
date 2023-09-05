#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import os

directory = r'C:\Users\DELL\Downloads\archive (7)\human_trafficking'

for dirname, _, filenames in os.walk(directory):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[7]:


get_ipython().system('pip install pycountry-convert')


# In[8]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pycountry_convert import country_alpha2_to_country_name, country_name_to_country_alpha3

# Importing warnings so that it may ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[10]:


df = pd.read_csv(r'C:\Users\DELL\Downloads\archive (7)\human_trafficking.csv')
df.head()


# In[11]:


df.shape


# In[12]:


df.columns.unique()


# In[13]:


df.isna()


# In[14]:


df.replace('-99', np.nan, inplace=True)
df.replace(-99, np.nan, inplace=True)

df.isna().head()


# In[15]:


df.head()


# In[16]:


def getCountryName(x):
    try:
        name = country_alpha2_to_country_name(x)
    except:
        name='Unknown'
    return name

def getCountryAlpha3(x):
    try:
        alpha3 = country_name_to_country_alpha3(x)
    except:
        alpha3 = 'Unknown'
    return alpha3


# In[17]:


df['country'] = df['citizenship'].apply(lambda x: getCountryName(x))
df['alpha3'] = df['country'].apply(lambda x: getCountryAlpha3(x))
df.head()


# In[18]:


df_map = pd.DataFrame(df.groupby(['country', 'alpha3'])['alpha3'].agg(Victims='count')).reset_index()
df_map.head()


# In[19]:


sns.set_theme(style='darkgrid', palette='pastel')
fig = plt.figure(figsize=(10, 5))
sns.barplot(df_map, x='country', y='Victims')
plt.xticks(rotation=90)
plt.show()


# In[20]:


world_map = px.choropleth(df_map, locations='alpha3', color='Victims', hover_name='country', 
                          color_continuous_scale='Sunsetdark', projection="natural earth")
world_map.update_layout(title_text='Human Trafficking Victims Worldwide')
world_map.show()


# In[21]:


df_victims = pd.DataFrame(df.groupby(['yearOfRegistration'])['yearOfRegistration'].agg(Victims='count')).reset_index()
df_victims


# In[22]:


sns.set_theme(style='darkgrid', palette='pastel')
fig = plt.figure(figsize=(10, 5))
ax = sns.barplot(df_victims, x='yearOfRegistration', y='Victims')
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 5), textcoords = 'offset points', fontsize=10)

plt.show()


# In[23]:


get_ipython().system('pip install pmdarima')


# In[24]:


get_ipython().system(' pip install statsmodels')


# In[25]:


from statsmodels.tsa.stattools import adfuller

df_test = adfuller(df_victims['Victims'], autolag='AIC')
print('ADF: ', df_test[0])
print('P-Values: ', df_test[1])
print('Num of Lags: ', df_test[2])
print('Num of Observations used fo calculating ADF Regression and Critical Values: ', df_test[3])
print('Critical Values: ', df_test[4])


# In[26]:


train = df_victims.iloc[:14, :]
test = df_victims.iloc[14:-1, :]
train


# In[27]:


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train['Victims'], order=(1,0,0))
model = model.fit()
model.summary()


# In[28]:


start = 14
end = 24

predictions = model.predict(start=start, end=end, typ='levels')
predictions


# In[29]:


years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2024, 2025]
plt.plot(df_victims['yearOfRegistration'], df_victims['Victims'], color='blue')
plt.plot(years, predictions, color='green')
plt.show()


# In[ ]:




