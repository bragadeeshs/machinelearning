#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter("ignore")


# ### Import the Dataset

# In[2]:


mydata = pd.read_csv("vehicle-1.csv")
mydata_copy = mydata.copy()


# ### Read the dataset

# In[3]:


mydata.head()


# ### Data pre-processing – Perform all the necessary preprocessing on the data ready to be fed to an Unsupervised algorithm

# ### Missing Values

# In[4]:


mydata.isnull().sum()


# ### Removal of missing values in dataset

# In[5]:


cols = ['compactness', 'circularity', 'distance_circularity', 'radius_ratio',
       'pr.axis_aspect_ratio', 'max.length_aspect_ratio', 'scatter_ratio',
       'elongatedness', 'pr.axis_rectangularity', 'max.length_rectangularity',
       'scaled_variance', 'scaled_variance.1', 'scaled_radius_of_gyration',
       'scaled_radius_of_gyration.1', 'skewness_about', 'skewness_about.1',
       'skewness_about.2', 'hollows_ratio']
for feature in cols:
    mydata[feature] = mydata[feature].fillna(mydata[feature].median())


# ### Confirmation of missing values

# In[6]:


mydata.isnull().sum()


# ### Check Outliers <br>
# 
# There are no outliers for Compactness<br>
# There are no outliers for Circularity<br>
# There are no outliers for distance_circularity<br>
# There are no outliers for scatter_ratio<br>
# There are no outliers for elongatedness<br>
# There are no outliers for pr.axis_rectangularity<br>
# There are no outliers for max.length_rectangularity<br>
# There are no outliers for scaled_radius_of_gyration<br>
# There are no outliers for skewness_about.2<br>
# There are no outliers for hollows_ratio<br>
# 
# There are outliers for radius_ratio<br>
# There are outliers for pr.axis_aspect_ratio<br>
# There are outliers for max.length_aspect_ratio<br>
# There are outliers for scaled_variance<br>
# There are outliers for scaled_variance.1<br>
# There are outliers for scaled_radius_of_gyration.1<br>
# There are outliers for skewness_about<br>
# There are outliers for skewness_about.1<br>
# 
# 

# In[7]:


for feature in cols:
    plt.figure(figsize = (6,6))
    mydata.boxplot([feature])


# ### Removal of Outliers with zscore
# 

# In[8]:


from scipy import stats
mydata['zradius_ratio'] = np.abs(stats.zscore(mydata['radius_ratio']))
mydata['zpr.axis_aspect_ratio'] = np.abs(stats.zscore(mydata['pr.axis_aspect_ratio']))
mydata['zmax.length_aspect_ratio'] = np.abs(stats.zscore(mydata['max.length_aspect_ratio']))
mydata['zscaled_variance'] = np.abs(stats.zscore(mydata['scaled_variance']))
mydata['zscaled_variance.1'] = np.abs(stats.zscore(mydata['scaled_variance.1']))
mydata['zscaled_radius_of_gyration.1'] = np.abs(stats.zscore(mydata['scaled_radius_of_gyration.1']))
mydata['zskewness_about'] = np.abs(stats.zscore(mydata['skewness_about']))
mydata['zskewness_about.1'] = np.abs(stats.zscore(mydata['skewness_about.1']))


# In[9]:


mydata.head()


# In[10]:


df_clean = mydata[mydata['zradius_ratio'] <= 3]
print("%i records have been removed after treating radius_ratio" %(mydata.shape[0]-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# In[11]:


df_clean = df_clean[df_clean['zpr.axis_aspect_ratio'] <= 3]
print("%i records have been removed after treating zpr.axis_aspect_ratio" %(record-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# In[12]:


df_clean = df_clean[df_clean['zmax.length_aspect_ratio'] <= 3]
print("%i records have been removed after treating zmax.length_aspect_ratio" %(record-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# In[13]:


df_clean = df_clean[df_clean['zscaled_variance'] <= 3]
print("%i records have been removed after treating zscaled_variance" %(record-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# In[14]:


df_clean = df_clean[df_clean['zscaled_variance.1'] <= 3]
print("%i records have been removed after treating zscaled_variance.1" %(record-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# In[15]:


df_clean = df_clean[df_clean['zscaled_radius_of_gyration.1'] <= 3]
print("%i records have been removed after treating zscaled_radius_of_gyration.1" %(record-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# In[16]:


df_clean = df_clean[df_clean['zskewness_about'] <= 3]
print("%i records have been removed after treating zskewness_about" %(record-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# In[17]:


df_clean = df_clean[df_clean['zskewness_about.1'] <= 3]
print("%i records have been removed after treating zskewness_about.1" %(record-df_clean.shape[0]))
record = df_clean.shape[0]
print("Total Records - %i" %(record))


# ### Copy to another dataframe

# In[18]:


mydata_copy = df_clean.copy()
a = df_clean.copy()


# ### Check Outliers after Treating with zscore<br>
# 
# There is a significant reduction in the outliers for all the columns

# In[19]:


col = ['zradius_ratio',
       'zpr.axis_aspect_ratio', 'zmax.length_aspect_ratio', 'zscaled_variance',
       'zscaled_variance.1', 'zscaled_radius_of_gyration.1', 'zskewness_about',
       'zskewness_about.1', 'class']


# In[20]:


df_clean.drop(col, axis = 1, inplace = True)


# In[21]:


for feature in df_clean.columns:
    plt.figure(figsize = (6,6))
    df_clean.boxplot(feature)


# ### Understanding the attributes - Find relationship between different attributes (Independent variables)

# In[22]:


col = ['zradius_ratio',
       'zpr.axis_aspect_ratio', 'zmax.length_aspect_ratio', 'zscaled_variance',
       'zscaled_variance.1', 'zscaled_radius_of_gyration.1', 'zskewness_about',
       'zskewness_about.1']
a.drop(col, axis = 1, inplace = True)


# In[23]:


a.describe(include = 'all').transpose()
#There are 3 unique values for the 'class' variable. Car has the highest count


# ### Correlation Values
# 
# We find that most of the independent variables are having a very high positive correlation<br>
# 
#    - Compactness has a very strong correlation with distance_circularity, circularity, radius_ratio, scatter_ratio, 	pr.axis_rectangularity<br>
#   - Circularity seems to be strongly correlated with scatter_ratio, pr.axis_rectangularity, max.length_rectangularity, scaled_variance, scaled_variance.1 and scaled_radius_of_gyration
#   - distance_circularity is strongly correlated with circularity, radius_ratio, scaled_variance, scaled_variance.1 <br>
#   - scatter_ratio is having a high correlation with compactness, circularity, distance_circularity, radius_ratio, pr.axis_rectangularity, max.length_rectangularity, scaled_variance, scaled_variance.1 and scaled_radius_of_gyration
#   - elongatedness is highly negatively correlated to scaled_variance and scaled_variance.1
#   - scaled_variance and scaled_variance.1 are almost correlated to 1
#   

# In[24]:


a.corr()


# ### Pairplot
# 
# From the pairplot, we get almost the same inferences from the correlation matrix. 

# In[25]:


sns.pairplot(a)


# ### Choose carefully which all attributes have to be a part of the analysis and why
# 
# We will drop the variables which has a correlation value of > 0.8 because one variable explains the other and there is no need to have both the variables in the same dataset. Here, we require domain experience as well to factor if we are missing out any relevant information while dropping the variables. Based on the correlation numbers, we decide to drop out the following: 
# 
#     elongatedness
#     pr.axis_rectangularity
#     max.length_rectangularity
#     scaled_radius_of_gyration
#     skewness_about.2
#     scatter_ratio
#     scaled_variance
#     scaled_variance.1
# 
# Note: Of course, there will be a higher accuracy if we include all the variables but it includes the noise generated by them as well. We need to strike a balance to not lose the information and on the other hand having minumum variables for the data set. This is a trade off to cater to the curse of dimentionality problem. Too many variables and lesser rows of data will lead to such a problem. 

# In[26]:


rem = ['max.length_rectangularity','scaled_radius_of_gyration','skewness_about.2','scatter_ratio','elongatedness','pr.axis_rectangularity',
    'scaled_variance','scaled_variance.1']
df_clean.drop(rem,axis = 1, inplace = True)


# ### Scaling of the data using Standard Scaler
# 
# Data needs to be scaled for any distance based algorithm such as clustering <br>

# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_clean = sc.fit_transform(df_clean)


# ### Finding Optimum number of Clusters
# 
# Cluster = 3 has the Silhoutte Score of 0.2399 and a good dip in average distortion

# In[28]:


from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

clusters = range(2,10)
meandistortion = []

for k in clusters:
    model = KMeans(n_clusters = k)
    model = model.fit(df_clean)
    prediction = model.predict(df_clean)
    meandistortion.append(sum(np.min(cdist(df_clean,model.cluster_centers_, 'euclidean'), axis = 1))/df_clean.shape[0])
    print("For Cluster = %i, the Silhouette Score is %1.4f" %(k,silhouette_score(df_clean,model.labels_)))
    
plt.plot(clusters, meandistortion, 'bx-')
plt.xlabel('k - Number of Clusters')
plt.ylabel('Average Distortion')
plt.title('Selecting k with the Elbow method')
    


# ### K Means Clustering Algorithm

# In[29]:


clus = KMeans(n_clusters = 3, random_state = 1)
clus.fit_predict(df_clean)


# In[30]:


col = ['zradius_ratio',
       'zpr.axis_aspect_ratio', 'zmax.length_aspect_ratio', 'zscaled_variance',
       'zscaled_variance.1', 'zscaled_radius_of_gyration.1', 'zskewness_about',
       'zskewness_about.1']
mydata_copy.drop(col,axis = 1, inplace = True)


# In[31]:


plt.scatter(df_clean[:,0], df_clean[:,1], c=clus.labels_)
plt.show()


# ### Hierarchial Clustering Method

# In[32]:


from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(df_clean, 'ward', metric = 'euclidean')
Z.shape


# In[33]:


plt.figure(figsize=(25, 10))
dendrogram(Z)
plt.show()


# In[34]:



dendrogram(Z,truncate_mode='lastp',p=3)
plt.show()


# In[35]:


from scipy.cluster.hierarchy import fcluster
max_d=40
clusters = fcluster(Z, max_d, criterion='distance')


# In[36]:


plt.scatter(df_clean[:,0], df_clean[:,1], c=clusters)  # plot points with cluster dependent colors
plt.show()


# ### Comparing the results of KMeans Cluster and Hierarchial Cluster 

# In[37]:


mydata_copy.head()


# In[38]:


mydata_copy['HCluster'] = clusters
mydata_copy['KCluster'] = clus.labels_
df_final = mydata_copy.copy()
df_final.drop('class', inplace = True, axis = 1)


# In[39]:


df_final.groupby('HCluster').median()


# In[40]:


df_final.groupby('KCluster').median()


# In[41]:


df_final['HC'] = np.where(df_final['HCluster'] == 1, 1, np.where(df_final['HCluster'] == 2, 2, 0))


# In[42]:


df_final['Match'] = np.where(df_final['KCluster'] == df_final['HC'], "True", "False")


# In[43]:


df_final.groupby('Match')['Match'].count()


# In[44]:


print("Accuracy percentage of match between KMeans and HCluster is %2.2f" %(100*df_final.groupby('Match')['Match'].count()[1]/df_final.shape[0]))


# ### Classification Technique - SVM

# In[45]:


c = mydata_copy.copy()
c.drop(['HCluster', 'KCluster'], axis = 1, inplace = True)
c.drop(rem, axis = 1, inplace = True)


# In[46]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c['class'] = le.fit_transform(c['class'])


# In[47]:



y = c['class']
X = c.drop(['class'], axis = 1)


# ### Test Train Data Split

# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size = 0.3)


# ### Data Scale

# In[49]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ### Applying the SVM Model

# In[50]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import metrics
svm = SVC(probability = True, random_state = 0)
svm.fit(X_train, y_train)
svmpredict = svm.predict(X_test)
print(classification_report(y_test, svmpredict))
print("Accuracy Score is %5.3f " %(accuracy_score(y_test, svmpredict) * 100))


# In[51]:


cm_svm = metrics.confusion_matrix(y_test, svmpredict, labels = [2,1,0])
df_cm_svm = pd.DataFrame(cm_svm, index = [i for i in ["2","1", "0"]], columns = [i for i in ["Predict 2", "Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm_svm, annot = True, cmap = "Greens", fmt='g')


# ### K Fold Cross Validation Score for SVM

# In[52]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svm, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ### Principal Component Analysis - Feature Extraction

# In[53]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 6, random_state = 0)
PCAX_train = pca.fit_transform(X_train)
PCAX_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_


# ### Variance Captured by PCA Components

# In[54]:


print(variance)


# ### Applying SVM Classification with PCA Components

# In[55]:


svmpca = SVC(probability = True, random_state = 0)
svmpca.fit(PCAX_train, y_train)
svmpredictpca = svmpca.predict(PCAX_test)
print(classification_report(y_test, svmpredictpca))
print("Accuracy Score is %5.3f " %(accuracy_score(y_test, svmpredictpca) * 100))


# In[56]:


cm_svm = metrics.confusion_matrix(y_test, svmpredictpca, labels = [2,1,0])
df_cm_svm = pd.DataFrame(cm_svm, index = [i for i in ["2","1", "0"]], columns = [i for i in ["Predict 2", "Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm_svm, annot = True, cmap = "Greens", fmt='g')


# ### K Fold Cross Validation Score

# In[57]:


accuraciespca = cross_val_score(estimator = svmpca, X = PCAX_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuraciespca.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuraciespca.std()*100))


# ### PCA Components wise model accuracies

# In[58]:



for components in range (2,10):
    pca = PCA(n_components = components, random_state = 0)
    PCAX_train = pca.fit_transform(X_train)
    PCAX_test = pca.transform(X_test)
    svmpca.fit(PCAX_train, y_train)
    svmpredictpca1 = svmpca.predict(PCAX_test)
    accuraciespca = cross_val_score(estimator = svmpca, X = PCAX_train, y = y_train, cv = 10)
    print("For %i Components:" %(components))
    print("Accuracy: {:.2f} %".format(accuraciespca.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuraciespca.std()*100))
    print()
    


# In[59]:


mean=[]
std=[]
for components in range (2,10):
    pca = PCA(n_components = components, random_state = 0)
    PCAX_train = pca.fit_transform(X_train)
    PCAX_test = pca.transform(X_test)
    svmpca.fit(PCAX_train, y_train)
    svmpredictpca1 = svmpca.predict(PCAX_test)
    accuraciespca = cross_val_score(estimator = svmpca, X = PCAX_train, y = y_train, cv = 10)
    mean.append(accuraciespca.mean()*100)
   
    


# ### Curve of PCA Components and Model Accuracy
# 
# Ideal PCA Components are 6

# In[60]:


plt.plot(range(2,10), mean, label = "Mean")
plt.xlabel("PCA Components")
plt.ylabel("Model Accuracy Percentage")
plt.title("PCA Components Vs Model Accuracy %")
plt.legend(loc = 4)
plt.show()


# ### Another method for PCA 

# In[61]:


sc = StandardScaler()
X_standard = sc.fit_transform(X)


# ### Construct a covariance matrix

# In[62]:


covariance_matrix = np.cov(X_standard.T)
print(covariance_matrix)


# ### Calculate the eigenvalues and eigenvectors

# In[63]:


eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print(eigenvalues)
print(eigenvectors)


# ### Form eigenpairs - Each pair will be an eigenvalue and the column wise eigenvector values

# In[64]:


eigenpair = [(eigenvalues[i], eigenvectors[:,i]) for i in range(len(eigenvalues))]


# In[65]:


print(eigenpair)


# ### Sort the eigenpair in descending order

# In[66]:


eigenpair.sort()
eigenpair.reverse()


# In[67]:


print(eigenpair)


# ### Separate the sorted eigenvalues and eigenvectors

# In[68]:


eigenvalues_sorted = [eigenpair[i][0] for i in range(len(eigenpair))]
eigenvectors_sorted = [eigenpair[i][1] for i in range(len(eigenpair))]


# ### Calculate the explained variance ratio to the sum of total variance

# In[69]:


total_variance = sum(eigenvalues_sorted)
for i in range(len(eigenvalues_sorted)):
    variance_explained = eigenvalues_sorted[i]/total_variance
    print(variance_explained)


# ### Add the cumulative sum of variance explained

# In[70]:


variance_explained = [(eigenvalues_sorted[i]/total_variance)for i in range(len(eigenvalues_sorted))]
cumulative_variance = np.cumsum(variance_explained)


# ### Draw a bar and step chart for visualization

# In[71]:



plt.bar(range(len(eigenvalues_sorted)), variance_explained, label = "Individual Variance Explained")
plt.step(range(len(eigenvalues_sorted)), cumulative_variance, label = "Cumulative Variance Explained")
plt.legend(loc = 'best')


# ### Convert the scaled original dataset and do a dot product to the eigenvectors. This will provide a transformed dataset

# In[72]:


PCAReduced = np.array(eigenvectors_sorted[0:6])
X_PCAReduced = np.dot(X_standard,PCAReduced.T)
df_PCAReduced = pd.DataFrame(X_PCAReduced)


# ### Do a pairplot with the PCA Transformed Dataset

# In[73]:


sns.pairplot(df_PCAReduced)


# ### Check if there is any correlation for the PCA Transformed Data set
# 
# There is no correlation and each independent variable is 'independent'

# In[74]:


df_PCAReduced.corr().round(5)


# ### Compare the accuracy scores and cross validation scores of Support vector machines – one trained using raw data and the other using Principal Components, and mention your findings

# ### SVM Without PCA and using Raw Data

# In[75]:


print(classification_report(y_test, svmpredict))
print("Accuracy Score is %5.3f " %(accuracy_score(y_test, svmpredict) * 100))


# ### SVM After PCA

# In[76]:


print(classification_report(y_test, svmpredictpca))
print("Accuracy Score is %5.3f " %(accuracy_score(y_test, svmpredictpca) * 100))


# ### Findings
# 
# SVM Before PCA
# - 10 Independent Variables were taken for the training and test set. We achieved an accuracy of 93.14%
# - Average Precision and Recall scores are 94% and 91% which is a good score. The F1 score is 93%. 
# - We find that Recall value is 75% for classifying correctly as 2 when the original data set value for the same row is 2. Out of 4 times, only once we are unable to classify them correctly. This is a measure of True Negatives. 
# 
# Now, when we do a PCA we are trying to eliminate the noise from the data and forming a new linear combination of independent variables into fewer variables. <br>
# The eigenvalue is the sum of maximum distance from the origin to the point where the data is projected onto the eigen vector. <br>
# The eigen vector when dot matrix multiplied to the original data set, we arrive at the PCA reduced dataset. <br>
# When we are reducing the number of variables, we can expect a reduction in the accuracy scores but at a benefit of handling lesser independent variables. <br>
# 
# SVM After PCA
# 
# - 6 PCA Components were taken for the training and test set. We achieved an accuracy of 86.69% after reducing to 6 components from 10 original independent variables
# - Average Precision and Recall scores are 86% and 84% which is a good score. The F1 score is 87%. We see that there is no big reduction in the F1 score. 
# - We find that Recall value is 61% for classifying correctly as 2 when the original data set value for the same row is 2. Out of 10 times, four times we are unable to classify them correctly. This is a measure of True Negatives. 
# 
# PCA does a good job and provides us a good accuracy score, when using production data we have to subject to the PCA reduction and then apply the classification algorithms. 
# 
# 

# In[ ]:




