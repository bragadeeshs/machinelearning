#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ## Import the data

# In[2]:


concrete = pd.read_csv("concrete.csv")


# ## Read the data

# In[3]:


concrete.head()


# ## Data shape, info and types
# There are 1030 rows and 9 columns. All are either float or int and no objects

# In[4]:


concrete.shape


# In[5]:


concrete.info()


# In[6]:


concrete.dtypes


# ## Find missing values
# 
# There are no missing values, but we need to find out from each column if there is any value with 0

# In[7]:


concrete.isnull().sum()


# In[8]:


zeroval = pd.DataFrame() #Create a DataFrame to store the number of records having '0' as the value
def zero(dataframe): #Create a function which returns the dataframe of records having '0' as the value
    for column in dataframe.columns:
        zeroval.set_value(0,column,dataframe[dataframe[column] == 0].shape[0])
        zeroval[column] = zeroval[column].astype('int64')
    return zeroval
zero(concrete)#Displays the dataframe with the column names and number of records with value '0'


# ### Find Median and fill the values 

# In[9]:


data = concrete.copy() #Create another copy of the dataframe


# In[10]:


medianval = pd.DataFrame() #Create a dataframe to store the median values of all columns
def median(dataframe): #Define a function which returns a dataframe with the median values
    for column in dataframe.columns:
        medianval.set_value(0,column,dataframe[column].median())
        medianval[column] = medianval[column].astype('float64')
    return medianval
median(data) #Displays the dataframe containing the median values


# In[11]:


def fillmedian(dataframe): #Create a function which returns the dataframe with the median values for values '0'
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].apply(lambda x: np.where(x == 0, dataframe[column].median(), x))
    return dataframe
fillmedian(data) #Displays the dataframe with the median values updated


# In[12]:


zero(data) #Check if there are still any column with value '0'


# In[13]:


data['ash'].plot(kind = 'hist') #We find that for 'ash', a very large number of records have '0' as value


# In[14]:


data[data['ash'] != 0]['ash'].plot(kind = 'hist') #Histogram with removing the value '0'


# In[15]:


data[data['ash'] != 0]['ash'].median() #Median value of 'ash' without considering the values '0'


# In[16]:


data['ash'] = data['ash'].apply(lambda x: np.where(x == 0, data[data['ash'] != 0]['ash'].median(), x))


# In[17]:


data['ash'].plot(kind = 'hist') #Histogram after applying the median transformation


# In[18]:


zero(data)#Recheck if there are any values with '0'. We dont find any and we are clear to go to EDA


# ## Univariate Analysis

# ### Box Plot
# 
# Cement & Courseagg independent variables have not outliers <br>
# Rest of the independent variables have outliers <br>

# In[19]:


plt.figure(figsize = (20,10))
data.boxplot()


# ### Histogram of all independent variables

# In[20]:


data.hist(figsize = (10,10))


# ### Correlation in the dataframe
# 
# Cement has 0.5 correlation with Strength <br>
# Superplastic and Age have almost 0.32 positive correlation with Strength <br>
# Cement, Slag, Superplastic & Age have a positive correlation with Strength. <br>
# Ash, Water, Coarseagg, Fineagg have a negative correlation with Strength <br>

# In[21]:


plt.figure(figsize = (20,10))
sns.heatmap(data.corr(), cmap = 'BrBG', annot = True)


# ### 5 Point Analysis
# Cement is evenly distributed as visualized through the histogram and boxplot<br>
# Slag has a max value of 359, but the median is at 22. The distribution is right skewed<br>
# Ash has been biased with median value as 121 after treating the values with '0'<br>
# Water looks like a normally distributed with a shorter range values than most other independent variables<br>
# Superplastic also like water seems to have a shorter range values <br>
# Coarseagg is normally distributed and does not have any outliers <br>
# Fineagg is also normally distributed and has outliers<br>
# Age variable does not have any distribution and have many discrete values<br>
# Strength is also normally distributed and lightly right skewed<br>

# In[22]:


data.describe(include = 'all').transpose().round()


# ### Distplot for all independent variables 
# 
# Cement has a normal distribution with one gaussian possibility<br>
# Slag possibly has two gaussians<br>
# Ash has possibly one gaussian<br>
# Water has possibly three gaussians<br>
# Superplastic has possibly two or three gaussians<br>
# Courseagg has possibly three gaussians<br>
# Fineagg has possibly three gaussians<br>
# Age possibly has five gaussians<br>
# Strength has a normal distribution with one gaussian possibility<br>

# In[23]:


fig, ax = plt.subplots(3,3,figsize = (15,15))
sns.distplot(data['cement'], ax = ax[0][0])
sns.distplot(data['slag'], ax = ax[0][1])
sns.distplot(data['ash'], ax = ax[0][2])
sns.distplot(data['water'], ax = ax[1][0])
sns.distplot(data['superplastic'], ax = ax[1][1])
sns.distplot(data['coarseagg'], ax = ax[1][2])
sns.distplot(data['fineagg'], ax = ax[2][0])
sns.distplot(data['age'], ax = ax[2][1])
sns.distplot(data['strength'], ax = ax[2][2])


# ## Multivariate Analysis

# ### Regplot for each independent variable to the dependent variable (Strength)
# 
# Cement has a positive correlation and the data points are packed and somewhat close to the best fit line<br>
# Slag has a positive correlation but the data points are spread out indicating a less correlation to Strength<br>
# Ash has a negative correlation but the data points are spread out indicating a less correlation to Strength<br>
# Water has a negative correlation but the data points are spread out indicating a less correlation to Strength<br>
# Superplastic has a positive correlation but the data points are packed out indicating a good correlation to Strength<br>
# Coarseagg has a negative correlation but the data points are spread out indicating a less correlation to Strength<br>
# Fineagg has a negative correlation but the data points are spread out indicating a less correlation to Strength<br>
# Age seems to have multiple discrete values and seems to be a non-linear fit<br>

# In[24]:


fig, ax = plt.subplots(2,4,figsize = (25,15))
sns.regplot(data['cement'], data['strength'], ax = ax[0][0])
sns.regplot(data['slag'], data['strength'], ax = ax[0][1])
sns.regplot(data['ash'], data['strength'], ax = ax[0][2])
sns.regplot(data['water'], data['strength'], ax = ax[0][3])
sns.regplot(data['superplastic'], data['strength'], ax = ax[1][0])
sns.regplot(data['coarseagg'], data['strength'], ax = ax[1][1])
sns.regplot(data['fineagg'], data['strength'], ax = ax[1][2])
sns.regplot(data['age'], data['strength'], ax = ax[1][3])


# ### Pairplot 
# 
# Superplastic and Water has a negative strong correlation <br>
# Cement and Strength has a very strong positive correlation<br>
# Fineagg and Water has a negative strong correlation<br>
# Age has a very poor correlation with the other independent variables<br>

# In[25]:


sns.pairplot(data, diag_kind = 'kde')


# ## Outlier processing
# 
# We will identify the outliers and replace them with the respective median values

# In[26]:


def outliers(dataframe): #Define a function for the outlier processing through the IQR method
    for column in dataframe.columns[:-1]:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5*iqr
        high = q3 + 1.5*iqr
        dataframe.loc[(dataframe[column] < low) | (dataframe[column] > high), column] = dataframe[column].median()
    return dataframe #Returns the dataframe after outlier processing through the IQR method
outliers(data)


# ### Boxplot after outlier processing
# 
# We find that most of the independent variables are free of outliers now. 

# In[27]:


plt.figure(figsize = (20,10))
data.boxplot()


# ## Feature Engineering
# 
# Q. Identify opportunities (if any) to create a composite feature, drop a feature etc.
# 
# We will do so using VIF and later in the code using PCA and also feature importance through Decision Tree

# ### Variance Inflation Factor
# 
# We find that "Ash" has a very big value of VIF and shows a high collinearity. We can discard this variable before our regression model is fit.

# In[28]:


def vif(dataframe):
    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    vif['Features'] = dataframe.columns
    return vif
vif(data)


# ### Drop Variables
# 
# We will drop "Ash" from the original dataset

# In[29]:


dataz = data.drop("ash", axis = 1)


# ### Explore for Gaussians by making clusters in the training data
# 
# Q. Explore for gaussians. If data is likely to be a mix of gaussians, explore individual clusters and present your findings in terms of the independent attributes and their suitability to predict strength
# 
# We will take n_clusters = 4

# In[30]:


def cluster(dataframe):
    clusters = range(2,10)
    mean_distortion = []
    for k in clusters:
        model = KMeans(n_clusters= k)
        model.fit(dataframe)
        prediction = model.predict(dataframe)
        mean_distortion.append(sum(np.min(cdist(dataframe, model.cluster_centers_, 'euclidean'), axis = 1))/dataframe.shape[0])
        print("For Cluster = %i, the Silhouette Score is %1.4f" %(k, silhouette_score(dataframe, model.labels_)))
    plt.plot(clusters, mean_distortion, 'bx-')
    plt.xlabel('k - Number of Clusters')
    plt.ylabel('Average Distortion')
    plt.title('Selecting K with Elbow method')
cluster(dataz)


# In[31]:


cluster = 4
def kmeans(dataframe, n_cluster):
    model = KMeans(n_clusters=cluster)
    model.fit(dataframe)
    prediction = model.predict(dataframe)
    return prediction
predictions = kmeans(dataz, cluster)


# In[32]:


def addcluster(dataframe, labels):
    dataframe['cluster'] = labels
    return dataframe
dataz = addcluster(dataz, predictions)


# In[33]:


dataz


# ### Make separate dataframes to each cluster formed
# 
# 
# 
# For each of the 4 clusters, we will seperate them to individual dataframes

# In[34]:


def dfclusters(dataframe):
    X_C0 = dataframe[dataframe['cluster'] == 0]
    X_C1 = dataframe[dataframe['cluster'] == 1]
    X_C2 = dataframe[dataframe['cluster'] == 2]
    X_C3 = dataframe[dataframe['cluster'] == 3]
    return X_C0, X_C1, X_C2, X_C3
data_C0, data_C1, data_C2, data_C3 = dfclusters(dataz)

print(data_C0.shape, data_C1.shape, data_C2.shape, data_C3.shape)


# ### Separate the independent and dependent variables

# In[35]:


columns=['strength', 'cluster']
def separate(dataframe):
    X = dataframe.drop(columns, axis = 1)
    y = dataframe['strength']
    return X, y
X_C0, y_C0 = separate(data_C0)
X_C1, y_C1 = separate(data_C1)
X_C2, y_C2 = separate(data_C2)
X_C3, y_C3 = separate(data_C3)


# ### Standardize the training, validation and test datasets

# In[36]:


def standardize(dataframe):
    dataframe = zscore(dataframe)
    return dataframe


# ### Creating a dataframe after Standardization

# In[37]:


def makedataframe(dataframe):
    X = pd.DataFrame(dataframe)
    X.columns = dataz.columns[:-2]
    return X


# In[38]:


X = dataz.columns[:-2]
y = ['strength']


# ### Data Split, apply models and returns the R2 score of the models in form of a dataframe
# 
# Q. Algorithms that you think will be suitable for this project<br>
# 
# We will use OLS, Lasso, Ridge, Decision Tree, Random Forest and Bagging Regressor models<br>

# In[39]:


def test(models, data, iterations = 10):
    results = {}
    for i in models:
        r2_train = []
        r2_test = []
        for j in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size = 0.3)
            r2_train.append(metrics.r2_score(y_train, models[i].fit(X_train, y_train).predict(X_train)))
            r2_test.append(metrics.r2_score(y_test, models[i].fit(X_train, y_train).predict(X_test)))  
        results[i] = [np.mean(r2_train), np.mean(r2_test)]
    return pd.DataFrame(results)


# In[40]:


models = {'OLS': linear_model.LinearRegression(),
         'Lasso': linear_model.Lasso(),
         'Ridge': linear_model.Ridge(),
         'DecisionTreeRegressor' : tree.DecisionTreeRegressor(),
         'RandomForestRegressor': ensemble.RandomForestRegressor(),
          'BaggingRegressor': ensemble.BaggingRegressor()
         }


# ### Results for each cluster from 0 to 3
# 
# 0 - Training Score <br>
# 1 - Testing Score <br>
# Random Forest algo does a good job considering average of 80% accuracy for all the clusters<br>

# In[41]:


test(models, data_C0)


# In[42]:


test(models, data_C1)


# In[43]:


test(models, data_C2)


# In[44]:


test(models, data_C3)


# ### Repeating the regression methods for the entire dataframe without spliting to clusters
# 
# 0 - Training score <br>
# 1 - Testing score<br>
# 
# Random Forest and Bagging Regressor has a good accuracy, but these are models without any hyperparameters tuned
# 

# In[45]:


test(models, dataz)


# ### Grid Search for all Regressors 
# 
# Q. Techniques employed to squeeze that extra performance out of the model without making it overfit or underfit
# 
# We will subject the data to hyperparameters and tune it to get the best performance

# In[46]:


lasso_params = {'alpha':[0.01, 0.1,1,10]}
ridge_params = {'alpha':[0.01, 0.1,1,10]}
tree_params = {'max_depth': [3,4,5]}
forest_params = {'max_depth': [3,4,5]}
bagging_params = {'max_features' : [3,4,5], 'n_estimators' : [10,25,50]}

models2 = {'OLS': linear_model.LinearRegression(),
           'Lasso': GridSearchCV(linear_model.Lasso(), 
                               param_grid=lasso_params).fit(dataz[X], dataz[y]).best_estimator_,
           'Ridge': GridSearchCV(linear_model.Ridge(), 
                               param_grid=ridge_params).fit(dataz[X], dataz[y]).best_estimator_,
            'DecisionTreeRegressor': GridSearchCV(tree.DecisionTreeRegressor(), 
                               param_grid=tree_params).fit(dataz[X], dataz[y]).best_estimator_,
            'RandomForestRegressor': GridSearchCV(ensemble.RandomForestRegressor(), 
                               param_grid=forest_params).fit(dataz[X], dataz[y]).best_estimator_,
           'BaggingRegressor': GridSearchCV(ensemble.BaggingRegressor(), 
                               param_grid=bagging_params).fit(dataz[X], dataz[y]).best_estimator_,
          }


# ### Accuracy after Hyperparameters Tuned
# We find that Random Forest and Bagging Regressor gives us a good result<br>

# In[47]:


test(models2, dataz)


# ### PCA
# 
# Transforming to 6 variables gives us the best score<br>

# In[48]:


A = dataz[X]
b = dataz['strength']

from sklearn.decomposition import PCA
def pca(components, dataframe):
    pca = PCA(n_components = components, random_state = 0)
    pca_X = pca.fit_transform(A)
    return pca_X
pca_A =  pca(6,A)


# In[49]:


pca_A = pd.DataFrame(pca_A)
pca_A = pca_A.join(b)


# In[50]:


C = pca_A.columns[:-1]
d = ['strength']


# ### After PCA, Data Split, apply models and returns the R2 score of the models in form of a dataframe

# In[51]:


def test1(models, data, iterations = 10):
    results = {}
    for i in models:
        r2_train = []
        r2_test = []
        for j in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(data[C], data[d], test_size = 0.3)
            r2_train.append(metrics.r2_score(y_train, models[i].fit(X_train, y_train).predict(X_train)))
            r2_test.append(metrics.r2_score(y_test, models[i].fit(X_train, y_train).predict(X_test)))  
        results[i] = [np.mean(r2_train), np.mean(r2_test)]
    return pd.DataFrame(results)


# ### Results without hypertuning done
# 
# Again Random Forest and Bagging Regressor gives us good results <br>

# In[52]:


test1(models, pca_A)


# ### Hyperparameter Tuning

# In[53]:


lasso_params = {'alpha':[0.01, 0.1,1,10]}
ridge_params = {'alpha':[0.01, 0.1,1,10]}
tree_params = {'max_depth': [3,4,5]}
forest_params = {'max_depth': [3,4,5]}
bagging_params = {'max_features' : [3,4,5], 'n_estimators' : [10,25,50]}

models3 = {'OLS': linear_model.LinearRegression(),
           'Lasso': GridSearchCV(linear_model.Lasso(), 
                               param_grid=lasso_params).fit(pca_A[C], pca_A[d]).best_estimator_,
           'Ridge': GridSearchCV(linear_model.Ridge(), 
                               param_grid=ridge_params).fit(pca_A[C], pca_A[d]).best_estimator_,
            'DecisionTreeRegressor': GridSearchCV(tree.DecisionTreeRegressor(), 
                               param_grid=tree_params).fit(pca_A[C], pca_A[d]).best_estimator_,
            'RandomForestRegressor': GridSearchCV(ensemble.RandomForestRegressor(), 
                               param_grid=forest_params).fit(pca_A[C], pca_A[d]).best_estimator_,
           'BaggingRegressor': GridSearchCV(ensemble.BaggingRegressor(), 
                               param_grid=bagging_params).fit(pca_A[C], pca_A[d]).best_estimator_,
          }


# ### Results after Hyper-Parameter Tuning is done
# 
# 0 - Training Score <br>
# 1 - Testing Score <br>
# Randon Forest and Bagging Regressor gives us a good result. 

# In[54]:


test1(models3, pca_A)


# ### Polynomial Regression
# 
# Q. Decide on complexity of the model, should it be simple linear model in terms of parameters or would a quadratic or higher degree help
# 
# We will subject the data to polynomial features and then apply the OLS, Lasso and the Ridge regression. 

# In[55]:


lasso_params = {'fit__alpha':[0.01, 0.1,1,10]}
ridge_params = {'fit__alpha':[0.01, 0.1,1,10]}

pipe1 = Pipeline([('poly', PolynomialFeatures()),
                 ('fit', linear_model.LinearRegression())])
pipe2 = Pipeline([('poly', PolynomialFeatures()),
                 ('fit', linear_model.Lasso())])
pipe3 = Pipeline([('poly', PolynomialFeatures()),
                 ('fit', linear_model.Ridge())])

models4 = {'OLS': pipe1,
           'Lasso': GridSearchCV(pipe2, 
                                 param_grid=lasso_params).fit(dataz[X], dataz[y]).best_estimator_ ,
           'Ridge': GridSearchCV(pipe3, 
                                 param_grid=ridge_params).fit(dataz[X], dataz[y]).best_estimator_,}


# ### Accuracy scores after Polynomial Features
# 
# 0 - Training Score<br>
# 1 - Testing Score<br>
# 
# Almost all the models are providing an accuracy of 81%

# In[56]:


test(models4,dataz)


# ### Finding the feature importance of the independent variables through Decision Tree

# In[57]:


dt = tree.DecisionTreeRegressor(max_depth = 5)
X, y = separate(dataz)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[58]:


X_trainstd = standardize(X_train)
X_teststd = standardize(X_test)


# In[59]:


X_trainstd = makedataframe(X_trainstd)
X_teststd = makedataframe(X_teststd)


# In[60]:


dt.fit(X_train, y_train)
pred = dt.predict(X_test)
print("Training Score is %2.2f" %(100 * dt.score(X_train, y_train)))
print("Testing Score is %2.2f" %(100 * dt.score(X_test, y_test)))


# ### Feature Importance
# 
# Q. Obtain feature importance for the individual features and present your findings
# 
# We find that Cement, Age, Water, Slag and Superplastic are important while FineAgg and Coarseagg are not<br>

# In[61]:


features = {}
for i in range(0,len(dt.feature_importances_)):
    features[i] = [dt.feature_importances_[i]]
features = pd.DataFrame(features)
features.columns = X_train.columns
features.sort_values(by = 0,axis = 1, ascending=False)


# ### Removing the non-important features and retrying the models

# In[85]:


X = ['cement', 'age', 'water', 'slag', 'superplastic']
y = ['strength']


# ### Accuracy after removing the non-important independent variables
# 
# We find that after removing two independent variables there is not much of an accuracy loss

# In[63]:


test(models2, dataz)


# ### Accuracy after removing the non-important independent variables and applying polynomial features 
# 
# We find that there is no big reduction in Accuracy loss after applying Polynomial Features and removing non-important features<br>

# In[64]:


test(models4,dataz)


# ### K Fold Validation and arriving at the average & stddev of the accuracy scores to find the model performance
# 
# Q. Model performance range at 95% confidence level
# 
# Model has an average accuracy of 75.432% and the range is [94.79165417269549, 85.44786495874915] since the standard deviation is 4.048% for 95% confidence interval

# In[86]:


X = dataz[X]
y = dataz['strength']
X = standardize(X)
X = pd.DataFrame(X)
scores = []
dt = tree.DecisionTreeRegressor(max_depth = 5)


# In[87]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[88]:


kfold = KFold(n_splits=10, random_state=1)
results = cross_val_score(dt, X, y, cv=kfold)
print(results)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[108]:


HigherRange = results.mean()*100 + 2*results.std()*100
LowerRange = results.mean()*100 - 2*results.std()*100
ModelRange = [HigherRange, LowerRange]
ModelRange

