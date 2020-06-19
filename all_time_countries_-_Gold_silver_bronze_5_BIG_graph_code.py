#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


plt.rcParams.update({'font.size': 22})     # setting font size in graph plotting

#if you want to see all the rows or columns of a dataframe
pd.set_option('display.max_columns', 100)  # or 1000
pd.set_option('display.max_rows', 100)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199, svm

#reading the csv files


# In[3]:


athlete_dataframe = pd.read_csv('athlete_events.csv')
region_dataframe = pd.read_csv('noc_regions.csv')

#merging the data frames into a single frame
merged_dataframe = pd.merge(athlete_dataframe,right=region_dataframe,on='NOC', how='left')
#first parameter is the orignal dataframe ; right gives the name of frame to merged with self ; how gives type of join left outer,right outer...
#print(merged_dataframe.head())

merged_dataframe.drop(['notes'],axis=1,inplace=True)
merged_dataframe['Medal'].fillna('No Medal',inplace=True)
merged_dataframe.head()


# In[ ]:


medals_tally = merged_dataframe[['region','Medal']][merged_dataframe.Medal!='No Medal']

medals_tally_total_count = medals_tally.groupby(['region']).count()
medals_tally_total_count.rename(columns={'Medal':'#MEDALS'},inplace=True)
medals_tally_total_count.reset_index(level=0, inplace=True)   # making the region index as column


GBS_table= medals_tally.groupby(['region','Medal']).Medal.count()
GBS_table=GBS_table.to_frame()              #GBS_table was a series 
GBS_table.rename(columns={'Medal':'#MEDALS2'},inplace=True)
GBS_table.reset_index(level=['region','Medal'], inplace=True)
GBS_table


# In[ ]:





# In[ ]:





# In[ ]:


# only run this one time (at the time of creating pickle )

# pkTally = open('pkTallyFile','ab')


# #Creating binary columns for medals\
# def gold_m(merged_dataframe)  :
#     if merged_dataframe["Medal"] == "Gold" :
#         return 1 
#     else :
#         return 0
# def silver_m(merged_dataframe)  :
#     if merged_dataframe["Medal"] == "Silver" :
#         return 1 
#     else :
#         return 0
# def bronze_m(merged_dataframe)  :
#     if merged_dataframe["Medal"] == "Bronze" :
#         return 1 
#     else :
#         return 0
    
# merged_dataframe["Gold"]   = merged_dataframe.apply(lambda x:gold_m(x),axis = 1) 
# merged_dataframe["Silver"] = merged_dataframe.apply(lambda x:silver_m(x),axis = 1) 
# merged_dataframe["Bronze"] = merged_dataframe.apply(lambda x:bronze_m(x),axis = 1) 
# merged_dataframe["Total"] = merged_dataframe["Gold"] + merged_dataframe["Silver"] +merged_dataframe["Bronze"]

# region_med = merged_dataframe.groupby("region")["Gold", 'Silver', 'Bronze',"Total"].sum().reset_index()


# pickle.dump( region_med,pkTally)
# pkTally.close()


# In[ ]:


pkTallyRead = open('pkTallyFile', 'rb')      
db = pickle.load(pkTallyRead)
db


# In[ ]:


y = (range(len(db)))
new_y = [i*15 for i in y] # for making values to be plotted spaced out (sparsely)
ht=6

new_y = np.array(new_y)          #you can perform elementwise operations on array (not required here unless you want +- something)
plt.figure(figsize=(50,500))
plt.barh(new_y-ht,db['Gold'],height=ht)
plt.barh(new_y,db['Silver'],height=ht)
plt.barh(new_y+ht,db['Bronze'],height=ht)


#plt.title('Medal count of '+country_name+' year-wise',fontsize=25)
plt.xticks(np.arange(0, 3000, 100),fontsize = 50,rotation=90)
# 0 is the initial value, 3000 is the final value (last value is not taken) and 100 is the difference of values between two consecutive ticks

plt.yticks(new_y,db['region'],fontsize= 50)
#manually changing yticks from new_x to db['region'] (i.e. country name)
plt.xlabel('med count',fontsize= 50)
plt.ylabel('country',fontsize=50)
#plt.legend(fontsize=25)
#plt.savefig('mt2.png') saves it perfectly
plt.show()


#in horizontal bar chart height is the width of each bar


# In[ ]:





# In[ ]:


new_y

# import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = "Crime Type Summer|Crime Type Winter".split("|")

# Generate dummy data into a dataframe
j = {x: [random.choice(["ASB", "Violence", "Theft", "Public Order", "Drugs"]
                       ) for j in range(300)] for x in s}
df = pd.DataFrame(j)

index = np.arange(5)
bar_width = 0.35

fig, ax = plt.subplots()
summer = ax.bar(index, df["Crime Type Summer"].value_counts(), bar_width,
                label="Summer")

winter = ax.bar(index+bar_width, df["Crime Type Winter"].value_counts(),
                 bar_width, label="Winter")

ax.set_xlabel('Category')
ax.set_ylabel('Incidence')
ax.set_title('Crime incidence by season, type')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["ASB", "Violence", "Theft", "Public Order", "Drugs"])
ax.legend()

plt.show()
# In[ ]:





# In[ ]:




