#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the dataframe, rename header assign the column names

# In[2]:


data = pd.read_csv("ratings_Electronics.csv", header = None, names = ['userid', 'productid','ratings', 'timestamp'])


# In[3]:


data.drop('timestamp', axis = 1, inplace = True)


# ## Data Characteristics

# In[4]:


data.head()


# In[5]:


data.dtypes


# In[6]:


data['userid'] = data['userid'].astype(str)
data['productid'] = data['productid'].astype(str)


# In[7]:


data.shape


# ### Check for missing values
# 
# No missing values

# In[8]:


data.isnull().sum()


# ### Ratings Histogram

# In[9]:


data['ratings'].plot(kind = 'hist')


# ### Ratings Countplot
# 
# We find that many of the items has received a rating of 5 from the users. 

# In[10]:


sns.countplot(data['ratings'])


# ### Unique Users and Products

# In[11]:


print('There are {} unique users'.format(len(data['userid'].unique())))
print('There are {} unique items'.format(len(data['productid'].unique())))


# ### Ratings Column Statistics
# 
# Mean rating is 4.012. Median rating is 5. Min rating is 1 and the max rating is 5. 

# In[12]:


data['ratings'].describe().transpose().round(3)


# # Create a dense, less sparse matrix
# 
# We will create a dataframe containing the Users with minimum 50 ratings

# In[13]:


data.groupby('userid').size().sort_values(ascending = False)[:100]


# In[14]:



df = data.copy()


# In[15]:


counts = df['userid'].value_counts()


# In[16]:


df = df[df['userid'].isin(counts[counts>=50].index)]


# In[17]:


print("Number of users who rated items or more are: {}".format(df.shape[0]))
print("Unique Users are {}".format(len(df['userid'].unique())))
print("Unique Items are {}".format(len(df['productid'].unique())))


# In[18]:


final = df.pivot(index = 'userid', columns = 'productid', values = 'ratings').fillna(0)


# In[19]:


print("Shape of final ratings matrix: ", final.shape)
number_of_ratings = np.count_nonzero(final)
print("Number of ratings given = ", number_of_ratings)
possible_ratings = final.shape[0]*final.shape[1]
print("Possible ratings = ", possible_ratings)
density = (number_of_ratings/possible_ratings)
density *=100
print("Density: {:4.2f}%".format(density))


# # Popularity Recommendation Model

# In[20]:


final_grouped = df.groupby('productid').agg({'userid': 'count'}).reset_index()
final_grouped.rename(columns = {'userid': 'score'}, inplace = True)


# In[21]:


final_grouped = final_grouped.sort_values(['score', 'productid'], ascending = [0,1])
final_grouped['rank'] = final_grouped['score'].rank(ascending = 0, method = 'first')


# ### Top 5 Recommendations

# In[22]:


final_grouped.head(5)


# # Split the data randomly into a train and test dataset.

# ### Using Surprise library, we will split the data and use for collaborative filtering approach

# In[23]:


from surprise import Dataset, Reader


# In[24]:


reader = Reader(rating_scale=(1,5))


# In[25]:


data = Dataset.load_from_df(df[['userid', 'productid','ratings']],reader)


# In[26]:


from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size= 0.30,random_state=1)


# # Collaborative Filtering Approach

# In[27]:


from surprise import KNNWithMeans
from surprise import accuracy


# In[28]:


algo = KNNWithMeans(k = 50, sim_options={'name':'pearson', 'user_based': True})
algo.fit(trainset)


# In[29]:


test_pred = algo.test(testset)


# ### RMSE for KNNwithMeans method

# In[30]:


accuracy.rmse(test_pred)
accuracy.mae(test_pred)


# ### Converting to a dataframe

# In[31]:


test_pred_df = pd.DataFrame(test_pred)


# In[32]:


test_pred_df['was_impossible'] = [x["was_impossible"] for x in test_pred_df['details']]


# In[33]:


test_pred_df


# ### Make Predictions

# In[34]:


algo.predict(uid='A19NP8YYADOOSF', iid = 'B00CGW74YU')


# ### Top 10 Predictions

# In[35]:


testset_new = trainset.build_anti_testset()


# In[36]:


predictions = algo.test(testset_new[0:1000])


# In[37]:


predictions_df = pd.DataFrame([[x.uid, x.iid,  x.est]for x in predictions])


# In[38]:


predictions_df.columns = ['userid', 'productid', 'est_rating']
predictions_df.sort_values(by = ['userid', 'est_rating'], ascending = False, inplace = True)


# In[39]:


predictions_df.groupby('userid').head(10).reset_index(drop = True)


# ## SVD Based Recommendation

# In[40]:


from surprise import SVD
from surprise import accuracy


# In[41]:


svd_model = SVD(n_factors=50,biased = False)
svd_model.fit(trainset)
test_pred_svd = svd_model.test(testset)


# ### RMSE for SVD 

# In[42]:


accuracy.rmse(test_pred_svd)
accuracy.mae(test_pred_svd)


# In[43]:


test_pred_svd[20]


# ### Parameter tuning for SVD

# In[44]:


from surprise.model_selection import GridSearchCV
param_grid = {'n_factors' : [5,10,15], "reg_all":[0.01,0.02]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3,refit = True)


# In[45]:


gs.fit(data)


# In[46]:


gs.param_combinations


# In[47]:


gs.best_params


# ### Buiding with best params and RMSE for SVD after Hyper Tuning

# In[48]:


svd_model = SVD(n_factors=5,reg_all = 0.02, biased = False)
svd_model.fit(trainset)
test_pred_svd = svd_model.test(testset)
accuracy.rmse(test_pred_svd)
accuracy.mae(test_pred_svd)


# In[49]:


test_pred_svd = pd.DataFrame(test_pred_svd)
test_pred_svd.head(10)


# ## Top 10 Recommendation from SVD

# In[50]:


testset_new = testset_new[:1000]
svd_pred = svd_model.test(testset_new)


# In[51]:


svd_pred_df = pd.DataFrame([[x.uid, x.iid, x.est] for x in svd_pred])
svd_pred_df.columns = ['userid', 'productid', 'est_rating']
svd_pred_df.sort_values(['userid', 'est_rating'], ascending = False, inplace = True)
svd_pred_df.groupby('userid').head(10).reset_index(drop = True)


# # Evaluate the Model & Insights

# KNNwithMeans - RMSE: 1.0465 and MAE:  0.7792<br>
# SVD without HyperParameter Tuning - RMSE: 2.0752 and MAE:  1.6771 <br>
# SVD after HyperParameter Tuning RMSE: 1.7340 and MAE:  1.3119 <br>
# <br>
# We find that KNNWithMeans is the best for User - User Collaborative model since it has the least RMSE and MAE <br>
# <br>
# While creating a dense matrix, we were able to achieve 0.17% only while taking the minimum number of ratings given by the user as 50 <br>
# We had the Unique Users are 1540 and Unique Items are 48190 <br>
# <br>
# Model-based Collaborative Filtering is a personalised recommender system, the recommendations are based on the past behavior of the user and it is not dependent on any additional information.<br>
# 
# The Popularity-based recommender system is non-personalised and the recommendations are based on frequecy counts, which may be not suitable to the user.<br>
# 
# The Popularity based model has recommended the same set of 5 products to both but Collaborative Filtering based model has recommended entire different list based on the user past purchase history<br>
# 

# In[ ]:




