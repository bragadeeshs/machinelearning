#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:



import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter("ignore")


# ### Load the data into a data-frame. The data-frame should have data and column description**

# In[2]:


df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df.head()


# ### Pandas Profiling

# In[3]:


import pandas_profiling
df.profile_report()


# ### Shape of Data
# 
# 5000 rows and 14 columns

# In[4]:


df.shape


# ### Ensure the attribute types are correct. If not, take appropriate actions.
# 
# Attributes are correct, all are either integer or float. None as object

# In[5]:


df.info()


# ### Presence of missing variables
# 
# There are no missing variables present in the dataframe

# In[6]:


df.isnull().sum()


# ### 5 point summary of numerical variable 
# 
# 1. Age - Youngest is 23 and eldest is 67 years. Average is 45 years
# 2. Experience average is 20 years
# 3. Income average is 74,000, minimum is 8000 and maximum is 224000.
# 4. Mortgage - Average is 56000, minimum is 0 and maximum is 635000. 
# 
# All Income and Mortgage figures in Dollars

# In[7]:


df.describe().transpose().round()


# ### Hypothesis Statements
# <ol>
#     <li>Customer with high salary will not opt for Personal Loan, Customer will Low and Medium Salary will opt for Personal Loan
#     <li>More family members for a customer, less chance for him/ her to opt for a Personal Loan
#     <li>Customers of the Age only less than 50 will opt for Personal Loan
#     <li>
# <ol>
# 
# Let us Test them !! 

# In[8]:


df['IncomeBins'] = pd.cut(x = df['Income'], bins = [0,20,40,60,80,100,120,140,160,180,200,220,240])
a = pd.crosstab(df['IncomeBins'], df['Personal Loan'])
a.apply(lambda a: 100*(a/a.sum())).plot(kind = 'bar', stacked = True)

#Customers with high salary (greater than $100000) have opted for Personal Loan


# In[9]:


a = pd.crosstab(df["Family"], df['Personal Loan'])
a.apply(lambda a: 100*(a/a.sum().round(2))).plot(kind = 'bar', stacked = True)

#Customer with 3, 4 family members have also opted for Personal Loan, in fact more than size of 1 or 2


# In[10]:


df['AgeBins'] = pd.cut(x = df['Age'], bins = [20,30,40,50,60,70])
a = pd.crosstab(df['AgeBins'], df['Personal Loan'])
a.apply(lambda a: 100*(a/a.sum())).plot(kind = 'bar', stacked = True)

#All the age groups, customers have responded to Personal Loan


# ## Univariate Analysis
# 
# ### Dist Plot for Age
# Almost Normally Distributed

# In[11]:


sns.distplot(df['Age'], color = "teal")


# ### Dist plot for Experience
# Almost Normally Distributed

# In[12]:


sns.distplot(df['Experience'], color = "teal")


# ### Dist plot for Income
# Right skewed distribution. Major of the income is earned below 50 years and slowly recedes after 50 years

# In[13]:


sns.distplot(df['Income'], color = "teal")


# ### Describe All Variables

# In[14]:


df.describe(include = 'all')


# ### Countplot of Family
# There are many customers with Family size of 1 

# In[15]:


sns.countplot(df['Family'])


# ### Dist plot of CCAvg
# Right Skewed distribution. Major of the spend is between 0 to 3000 in dollars

# In[16]:


sns.distplot(df['CCAvg'])


# ### Count plot of Education
# Majority of the customers are UnderGrad. Almost similar number of 1250-1500 range for Graduate and Advanced/Professional

# In[17]:


sns.countplot(df['Education'])


# ### Dist plot of Mortgage
# 
# Removed the data where Mortgage = 0 and then made a dist plot. 
# Data is right skewed, with major of the mortgage is around 80000 - 120000 in dollars

# In[18]:


mortgage = df[df['Mortgage']>0]
sns.distplot(mortgage['Mortgage'])


# ### Countplot of Personal Loan
# Majority of them have not accepted the personal loan that was offered to them in the earlier campaign

# In[19]:


sns.countplot(df['Personal Loan'])


# ### Value Count of Personal Loan 
# 9.6% of them accepted the personal loan that was offered to them in the earlier campaign

# In[20]:


df['Personal Loan'].value_counts(normalize = True)


# ### Countplot & Value Count of Securities Account
# Majority of them does not have a securities account with the bank
# 89.56% of them do not have a security account in bank where as 10.44% of them do have. 

# In[21]:


sns.countplot(df['Securities Account'])

df['Securities Account'].value_counts(normalize = True)


# ### Countplot and Value Counts for CD Account
# Majority of them do not have a certificate of deposit (CD) account with the bank
# 93.96% of the people do not have certificate of deposit (CD) account with bank, where as 6.04% of them do have. 

# In[22]:


sns.countplot(df['CD Account'])
df['CD Account'].value_counts(normalize = True)


# ### Countplot and Value Counts for Online Banking Facility
# 
# Majority of them do not have the Online Banking facility.
# 59.68% of the people have Online Banking Facility with the Bank, where as 40.32% of them do not have the facility

# In[23]:


sns.countplot(df['Online'])
df['Online'].value_counts(normalize = True)


# ### Count plot and Value Count for Credit Card
# Majority of the customers do not use credit card issued by UniversalBank. 
# 70.6% of the customers do not use credit card issued by Universal Bank, where as 29.4% of them do use. 

# In[24]:


sns.countplot(df['CreditCard'])
df['CreditCard'].value_counts(normalize = True)


# ## Multivariate Analysis
# 
# ### Joint plot for Age and Experience
# 
# There is strong positive co-relation between Age and Experience, almost equal to 1. 

# In[25]:


sns.jointplot(df['Age'], df['Experience'])


# ### Pairplot for Selected Features
# 
# <ol>
#     <li>There is a positive relationship between Income and CCAvg spend
#     <li>Age and Experience has a nearly perfect positive corelation
#     <li>Income < $50000 have not responded to Personal Loan 
# <ol>

# In[26]:


sns.pairplot(df, vars = ['Age', 'Experience', 'Income', 'CCAvg'], hue = 'Personal Loan')

#I have not applied pairplot for all the variables as the information gets messy. Hence applied to only key continuous variables 


# ### Crosstab between Family and Education
# 
# Customer with Family size as 1 and 2 are educated more as UnderGrad
# Customer with Family size as 3 and 4 are educated more as Graduate

# In[27]:


ct = pd.crosstab(df['Family'], df['Education'])
ct.plot(kind = 'bar')
ct.apply(lambda e: 100*(e/e.sum().round(2)))


# ### Divide Age into Bins and apply Crosstab to Family
# <ol>
#     <li>Customers in the Age group 40-50 are mostly of the Family size 1
#     <li>Customers in the Age group 20-30 are mostly of the Family size 4
# <ol>    

# In[28]:


df['AgeBins'] = pd.cut(x = df['Age'], bins = [20,30,40,50,60,70])
ct = pd.crosstab(df['AgeBins'], df['Family'])
ct.plot(kind = 'bar')
ct.apply(lambda e: 100*(e/e.sum()))


# ### Divide Mortgage into Bins and apply Crosstab to AgeBins
# <ol>
#     <li>Customers in the Age group 50-60 have the most cases of Mortgages ranging from $0 - $100000
#     <li>Customers in the Age group 30-40 have the most cases of Mortgages ranging from $50000 - $100000
# <ol>
# 
# All figures in dollars

# In[29]:


df['MortgageBins'] = pd.cut(x = df['Income'], bins = [0,50,100,150,200,250,300,350,400,450,500,550,600,650])
ct = pd.crosstab(df['AgeBins'], df['MortgageBins'])
ct.plot(kind = 'bar')
ct.apply(lambda e: 100*(e/e.sum()))


# ### Divide Income into Bins and Crosstab to AgeBins
# <ol>
#     <li>Among the Income Range between 20000-80000, the highest Age Range is 50-60
#     <li>Among the Income Range between 100000-160000, the highest Age Range is 40-50
# <ol>
# Income figures are in dollars

# In[30]:


df['IncomeBins'] = pd.cut(x = df['Income'], bins = [0,20,40,60,80,100,120,140,160,180,200,220,240])
ct = pd.crosstab(df['IncomeBins'], df['AgeBins'])
ct.plot(kind = 'bar')
ct.apply(lambda e: 100*(e/e.sum()))


# ## Target Column Distribution - In our case 'Personal Loan' is Target Variable
# 
# ### Countplot of Personal Loan
# 
# 90.4% of the customers have not responded to Personal Loan
# 9.6% of the customers have responded to Personal Loan

# In[31]:


sns.countplot(df['Personal Loan'])
df['Personal Loan'].value_counts(normalize = True)*100


# ## Data Distribution of 'Personal Loan' across all variables
# 
# ### Crosstab of Education to Personal Loan
# 
# Education 1: Undergrad - 19.37% opted, 2: Graduate - 37.91% and 3: Advanced/Professional - 42.7% have opted for the personal loan

# In[32]:


edu_crosstab = pd.crosstab(df['Education'], df['Personal Loan'])
edu_crosstab.apply(lambda a: (a/a.sum())*100, axis = 0)


# ### Crosstab of Family to Personal Loan
# <ol>
#     <li>Family size of 1, 2 have opted to almost same 22% to Personal Loan. 
#     <li>Family size of 3, 4 have opted to almost 28% to Personal Loan. 
# <ol>

# In[33]:


fam_crosstab = pd.crosstab(df['Family'], df['Personal Loan'])
fam_crosstab.apply(lambda b: (b/b.sum())*100, axis = 0)


# ### Crosstab of AgeBins to Personal Loan
# 
# We find that highest % of response is of the Age Group 30-35 with 14.1% and followed by 25-30 Age Group with 13.75%

# In[34]:


df['AgeBins'] = pd.cut(x = df['Age'], bins = [20,25,30,35,40,45,50,55,60,65,70])
agebins_crosstab = pd.crosstab(df['AgeBins'], df['Personal Loan'])
agebins_crosstab.apply(lambda d: 100*(d/d.sum()))


# ### Another perspective - Cut the bins in 10 years
# 
# We find that highest of 25.41% of the response to Personal Loan was from the Age Group 40-50. Also close were 30-40 and 50-60 Age Groups

# In[35]:


df['AgeBins'] = pd.cut(x = df['Age'], bins = [20,30,40,50,60,70])
agebins_crosstab = pd.crosstab(df['AgeBins'], df['Personal Loan'])
agebins_crosstab.apply(lambda d: 100*(d/d.sum()))


# ### Crosstab of CC Usage with Personal Loan
# <ol>
#     <li>People with CC Spending average between 2000 - 5000 have given the max response to Personal Loan (51.35%)
#     <li>People with CC Spending average between 5000 - 8000 have given the next max response to Personal Loan(25.88%)
#     <li>People with CC Spending average between 8000 - 10000 have given the least response to Personal Loan (2.92%)
# <ol>
# Spending Average figures are in dollars

# In[36]:


df['CCBins'] = pd.cut(x = df['CCAvg'], bins = [0,2,5,8,10])
cc_crosstab = pd.crosstab(df['CCBins'], df['Personal Loan'])
cc_crosstab.apply(lambda e: 100*(e/e.sum()))


# ### Crosstab of Income to Personal Loan
# <ol>
#     <li>People with Income < 40000 have not responded to Personal Loan
#     <li>People with Income > 200000 have not responded to Personal Loan
#     <li>People with Income in the range 100000 - 200000 have responded the most to Personal Loan
#     <li>People with Income in the range 120000 - 140000 have responded the most (22.29%)
# <ol>
# All income figures are in dollars

# In[37]:


df['IncomeBins'] = pd.cut(x = df['Income'], bins = [0,20,40,60,80,100,120,140,160,180,200,220,240])
cc_crosstab = pd.crosstab(df['IncomeBins'], df['Personal Loan'])
cc_crosstab.apply(lambda e: 100*(e/e.sum()))


# ### Crosstab of Security Deposit to Personal Loan
# <ol>
#     <li>87.5% of people who do not have Security Deposit Account has responded to Personal Loan
#     <li>12.5% of people who has a Security Deposit Account has responded to Personal Loan
# <ol>

# In[38]:


sec_crosstab = pd.crosstab(df['Securities Account'], df['Personal Loan'])
sec_crosstab.apply(lambda f: 100*(f/f.sum()), axis = 0)


# ### Crosstab of Mortgage Bins to Personal Loan
# <ol>
#     <li>People with Mortgage of 100000 - 150000 has 45.83% response to Personal Loan.
#     <li>People with Mortgage of 150000 - 200000 has 44.79% response to Personal Loan.
#     <li>Very low response to Personal Loan with Mortgage value of < $100000 and >$200000
# <ol>

# In[39]:


df['MortgageBins'] = pd.cut(x = df['Income'], bins = [0,50,100,150,200,250,300,350,400,450,500,550,600,650])
mortgage_crosstab = pd.crosstab(df['MortgageBins'], df['Personal Loan'])
mortgage_crosstab.apply(lambda e: 100*(e/e.sum()))


# ### Crosstab of CD Account to Personal Loan
# <ol>
#     <li>70.83% of people who do not have CD Account has responded to Personal Loan
#     <li>29.16% of people who has a CD Account has responded to Personal Loan
# <ol>

# In[40]:


cd_crosstab = pd.crosstab(df['CD Account'], df['Personal Loan'])
cd_crosstab.apply(lambda e: 100*(e/e.sum()))


# ### Crosstab of Online Banking to Personal Loan
# <ol>
#     <li>39.37% of people who do not have Online Banking Facility has responded to Personal Loan
#     <li>60.62% of people who has a Online Banking Facility has responded to Personal Loan
# <ol>

# In[41]:


online_crosstab = pd.crosstab(df['Online'], df['Personal Loan'])
online_crosstab.apply(lambda e: 100*(e/e.sum()))


# ### Crosstab of CC Usage to Personal Loan
# <ol>
#     <li>70.20% of people who do not have Credit Card Usage has responded to Personal Loan
#     <li>29.79% of people who has a Credit Card Usage has responded to Personal Loan
# <ol>

# In[42]:


cc_crosstab = pd.crosstab(df['CreditCard'], df['Personal Loan'])
cc_crosstab.apply(lambda e: 100*(e/e.sum()))


# ## Strategies to address the different data challenges such as data pollution, outliers and missing values.
# 
# ### Presence of missing values
# 
# Insights - There are no missing variables present in the dataframe. Education has negative values but I will drop the columns since Age and Experience gives same data due to almost a perfect corelation. Hence, will not focus on Education column. 

# In[43]:


df.isnull().sum()


# ### Outlier Treatment
# 
# No outliers present for Age
# No outliers present for Experience
# Outliers are present and found to be greater than 180000 dollars for Income variable

# In[44]:


sns.boxplot(df['Age'])


# In[45]:


sns.boxplot(df['Experience'])


# In[46]:


sns.boxplot(df['Income'])


# Outliers are present for Mortgage values > 250000 dollars. This could be because of many values of 0 dollars
# 69.24% of the people have 0 dollars as Mortgage and hence we have many outliers. 
# We need to treat this variable for outliers

# In[47]:


sns.boxplot(df['Mortgage'])
(df[df['Mortgage'] == 0]['Mortgage'].value_counts()/df.shape[0])*100


# Presence of outliers for CC Avg values more than > 5000 dollars and there is a need to treat the outliers

# In[48]:


sns.boxplot(df['CCAvg'])


# <ol>
#     <li>Outlier Treatment needs to be done for Mortgage
#     <li>We will employ Standardization technique with Z score. Any z score > 3 can be considered as outliers and dropped
# <ol>
#     Note: I have not done the outlier treatment to Income and CCAvg as the accuracy is reduced without these data points
# 

# In[49]:


from scipy import stats
df['Mortgage_Zscore'] = np.abs(stats.zscore(df['Mortgage']))
df_clean = df[df['Mortgage_Zscore'] < 3]
df_clean.shape
#Removed 105 rows and shape is now 4895 rows


# Idea for outlier treatment is that we are standardizing the values to a normal distribution and based on the 
# z score we will know these values are at how much standard deviations from the mean. I have chosen z value of 3
# which is close to 3 std dev from mean. So I have removed all the values which are > z score of 3. In effect
# I have removed 105 rows after outlier treatment

# ### Removing all unwanted/ newly created columns

# In[50]:


df_clean.head()
cols = ['Experience','ID', 'ZIP Code', 'AgeBins', 'MortgageBins', 'IncomeBins', 'CCBins', 'Mortgage_Zscore']
df_clean = df_clean.drop(cols, axis = 1)


# ### Split Data to target variable and independent variables

# In[51]:


df_clean.shape


# In[52]:


X = df_clean.drop(['Personal Loan'], axis = 1)
y = df_clean[['Personal Loan']]


# ### Create the training set and test set in ration of 70:30

# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### Transform the data i.e. scale / normalize if required

# In[54]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
#Separated into train and test datasets and then applied the Standard Scaler.
#This is a best practice since the algorithm will not be cognizant of the test data while training


# ### First create Logistic Regression algorithm. Note the model performance.
# 
# ### Create Roc Curve and calculate the fpr and tpr for all thresholds of the classification

# In[55]:


#Import the libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import metrics

#Fit the model and Predict

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

#Classification Report and Accuracy Score
print(classification_report(y_test, y_pred_lr))
print("Accuracy Score is %5.3f " %(accuracy_score(y_test, y_pred_lr) * 100))

#ROC Curve for Logistic Regression
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = round(metrics.roc_auc_score(y_test, y_pred_proba)*100,2)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.title("ROC for Logistic Regression")
plt.legend(loc=4)
plt.show()

#Confusion Matrix for Logistic Regression
cm_lr = metrics.confusion_matrix(y_test, y_pred_lr, labels = [1,0])
df_cm_lr = pd.DataFrame(cm_lr, index = [i for i in ["1", "0"]], columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm_lr, annot = True, cmap = "Greens", fmt='g')

#Recall value is 59%, Precision is 82% and F1 score is 69% for predicting True Positives of 1. 
#We need to find out the Recall value for other models. 
#Accuracy is good - 95.30


# ### Logistic Regression with Hyper Parameter Tuning using GridSearchCV
# 
# ### Print Best accuracy and Best Parameters using Confusion Matrix & Print Classification Report to check precision, recall & F1 Score!

# In[56]:


from sklearn.model_selection import GridSearchCV
grid_params = {"C": [1,10,100], "penalty":['l2'], "solver": ['liblinear', 'saga', 'lbfgs', 'sag']}
logreg = LogisticRegression()
model_grid = GridSearchCV(logreg, grid_params, cv = 10)
model_grid.fit(X_train, y_train)
y_pred_lr_grid = model_grid.predict(X_test)

#Accuracy Score and Classification Report
print("Accuracy Score is %5.3f" %(accuracy_score(y_test, y_pred_lr_grid)*100))
print(classification_report(y_test, y_pred_lr_grid))

#Best Parameters and Score for Grid Search
print("Best Parameters for Logistic Regression after Grid Search is %s" %(model_grid.best_params_))
print("Best Score of Logistic Regression - Grid Search is %5.3f" %(model_grid.best_score_ * 100))
#Best score is 95.42. This is best of scores out of 10 cross validations

#ROC Curve
y_pred_proba_grid = model_grid.predict_proba(X_test)[::,1]
fpr_lr,tpr_lr,_=metrics.roc_curve(y_test, y_pred_proba_grid)
auc = round(metrics.roc_auc_score(y_test,y_pred_proba_grid)*100,2)
plt.plot(fpr_lr,tpr_lr, label = "auc="+str(auc))
plt.legend(loc = 4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Logistic Grid Search")
plt.show()

#Confusion Matrix for Grid Search - Logistic Regression
cm_lrgrid = metrics.confusion_matrix(y_test, y_pred_lr_grid, labels = [1,0])
df_cm_lrgrid = pd.DataFrame(cm_lrgrid, index = [i for i in ["1", "0"]], columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm_lrgrid, annot = True, cmap = "Greens", fmt='g')
      
#Grid Search shows the same results as the Logistic Regression because the best params are the default params of LR


# ### Build KNN algorithm and explain why that algorithm in the comment lines.
# 
# 1. KNN is a supervised algorithm, it is non-parametric and lazy (instance-based). 
# 2. Training is it does not explicitly learn the model, but it saves all the training data and uses the whole training set for classification or prediction
# 3. Training process is very fast, it just saves all the values from the data set
# 4. Very useful algorithm in case of small data sets 

# In[57]:


#Import Libraries
from sklearn.neighbors import KNeighborsClassifier

#Fit and Predict KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_predict_knn = knn.predict(X_test)

#Print Accuracy Score and Classification Report
print("Accuracy Score is %5.3f" %(accuracy_score(y_test,y_predict_knn) * 100))
print(classification_report(y_test, y_predict_knn))

#ROC Curve for KNN Model
y_predict_proba_knn = knn.predict_proba(X_test)[::,1]
fpr, tpr,_=metrics.roc_curve(y_test, y_predict_proba_knn)
auc=round(metrics.roc_auc_score(y_test,y_predict_proba_knn)*100,2)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve KNN Classifier")
plt.legend(loc=4)
plt.show()

#Confusion Matrix for KNN Model
cm_knn = metrics.confusion_matrix(y_test, y_predict_knn, labels = [1,0])
df_cm_knn = pd.DataFrame(cm_knn, index = [i for i in ["1", '0']], columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm_knn, annot = True, cmap = "Greens", fmt = 'g')

#Recall is 66%, Precision is 94% and F1 score is 78%. This is better than LR. 
#We are able to correctly classify 66% of the times the predicted value as 1 when the actual value is 1 
#Accuracy Score is 96.73, this is better than LR


# ### K-Nearest Neighbour with Hyper Parameter Tuning using GridSearchCV. Using Grid Search to identify optimum value for K.
# 
# ### Print Best accuracy and Best Parameters using Confusion Matrix & Print Classification Report to check precision, recall & F1 Score	

# In[58]:


#Define the Grid Parameters for KNN
grid_param_knn = {"n_neighbors": [3,5,7,9], "weights": ['uniform', 'distance'], "metric": ['euclidean', 'manhattan']}

#Fit and Predict with KNN Model
knn = KNeighborsClassifier()
grid_model_knn = GridSearchCV(knn, grid_param_knn, cv = 10)
grid_model_knn.fit(X_train, y_train)
y_predict_knn_grid = grid_model_knn.predict(X_test)

#Accuracy Score and Classification Report
print('Model Accuracy Score is %5.2f' %(100*accuracy_score(y_test, y_predict_knn_grid)))
print(classification_report(y_test, y_predict_knn_grid))

#KNN Grid Search Best Score and Parameters
print('Grid Search Best Score for KNN %5.2f' %(100 * grid_model_knn.best_score_))
print('Grid Search Best Params for KNN %s' %(grid_model_knn.best_params_))

#ROC Curve for KNN Grid Search
y_predict_proba_knngrid = grid_model_knn.predict_proba(X_test)[::,1]
fpr_knn, tpr_knn,_ = metrics.roc_curve(y_test, y_predict_knn_grid)
auc = round(metrics.roc_auc_score(y_test, y_predict_knn_grid)*100,2)
plt.plot(fpr_knn,tpr_knn,label="auc="+str(auc))
plt.legend(loc = 4)
plt.title("KNN Grid ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

#Confusion Matrix for KNN Grid Search
cm_knn_grid = metrics.confusion_matrix(y_test, y_predict_knn_grid, labels = [1,0])
df_cm_knn_grid = pd.DataFrame(cm_knn_grid, index = [i for i in ["1", '0']], columns = [i for i in ["Predict 1", "Predict 0"]])
sns.heatmap(df_cm_knn_grid, annot = True, fmt = 'g', cmap = "Greens")

#After Grid Search, Precision has increased to 95%. Rest of the values are more or less same as normal KNN
#Default n_neighbors value is 5 for KNN, best params of Grid search defines it as 3. 
#Accuracy is 96.73%, which is same as normal KNN model


# ### KNN - Optimum k value using Elbow Curve
# 
# As per elbow curve, the optimum neighbors to query is 5 neighbors

# In[59]:


scores = []
for k in range(1,10):
    elbow = KNeighborsClassifier(n_neighbors = k)
    elbow.fit(X_train, y_train)
    elbow.predict(X_test)
    scores.append(elbow.score(X_test, y_test))
plt.plot(range(1,10), scores)

#Elbow curve shows the n_neighbors should be 5 with an accuracy of 97.5% approx


# ### Rebuild the KNN using the optimum value to achieve best accuracy
# 
# ### Print Best accuracy and Best Parameters using Confusion Matrix & Print Classification Report to check precision, recall & F1 Score	
# 
# {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'} parameters are used in KNN Model

# In[60]:


#Fit and Predict KNN model
knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 3, weights = 'distance')
knn.fit(X_train, y_train)
y_predict_knn = knn.predict(X_test)

#Print Accuracy Score and Classification Report
print("Accuracy Score is %5.3f" %(accuracy_score(y_test,y_predict_knn) * 100))
print(classification_report(y_test, y_predict_knn))

#ROC Curve for KNN Model
y_predict_proba_knn = knn.predict_proba(X_test)[::,1]
fpr, tpr,_=metrics.roc_curve(y_test, y_predict_proba_knn)
auc=round(metrics.roc_auc_score(y_test,y_predict_proba_knn)*100,2)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve KNN Classifier")
plt.legend(loc=4)
plt.show()

#Confusion Matrix for KNN Model
cm_knn = metrics.confusion_matrix(y_test, y_predict_knn, labels = [1,0])
df_cm_knn = pd.DataFrame(cm_knn, index = [i for i in ["1", '0']], columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm_knn, annot = True, cmap = "Greens", fmt = 'g')

#Precision, Recall and F1 score has been explained in the previous cell


# ### Build Naive Bayes Classifier on training Dataset and explain why that algorithm in the comment lines
# 
# 1. Bayes theorem uses the conditional probability of an event. Events should be mutually exclusive like throwing a dice.
# 2. Bayes Theorem assumes predictors or input features are independent of each other.
# 3. Bayesian probability relates to the degree of belief. It gives the likelihood of an event to occur. It does this with prior knowledge of the condition related to the event
# 
# ### Print the accuracy of the model & confusion Matrix for Naïve Bayes Model.
# 
# ### Explain Precision, Recall value & F1 Score using the classification report
# 
# ### Calculate the fpr and tpr for all thresholds of the classification.
# 
# ### Firstly, calculate the probabilities of predictions made & then plot the ROC Curve

# In[61]:


#Import the libraries
from sklearn.naive_bayes import GaussianNB

#Fit and Predict with NB Model
nb = GaussianNB()
nb.fit(X_train, y_train)
y_predict_nb = nb.predict(X_test)

#Accuracy Score and Classification Report
print("Accuracy Score is %5.3f" %(accuracy_score(y_test, y_predict_nb)*100))
print(classification_report(y_test, y_predict_nb))

#ROC for Naive Bayes
y_predict_proba_nb = nb.predict_proba(X_test)[::,1]
fpr_nb,tpr_nb,_ = metrics.roc_curve(y_test, y_predict_proba_nb)
auc = round(metrics.roc_auc_score(y_test, y_predict_proba_nb)*100,2)
plt.plot(fpr_nb,tpr_nb,label = "auc="+str(auc))
plt.title("ROC for Naive Bayes")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc = 4)
plt.show()

#Confusion Matrix for Naive Bayes
cm_nb = metrics.confusion_matrix(y_test, y_predict_nb, labels = [1,0])
df_cm_nb = pd.DataFrame(cm_nb,index = [i for i in ["1", "0"]], columns = [i for i in ["Predict 1", "Predict 0"]])
sns.heatmap(df_cm_nb, annot = True, fmt = 'g', cmap = "Greens")

#Recall is 59%, Precision is 42% and F1 score is 50%. 
#Naive Bayes is a very bad model for this dataset. 
#Accuracy is 89.44%, though decent, the precision gets hit very bad. The overall F1 score reduces drastically. 


# ### SVM Classifier Model
# 
# Accuracy Score is 97.68, Precision is 0.90, Recall is 0.83 and F1 score is 0.86.
# This model is the best among all the other classifiers

# In[62]:


from sklearn import svm
svm_model = svm.SVC(probability = True, C = 10, gamma = 0.1, kernel = 'rbf')
svm_model.fit(X_train, y_train)
y_predict_svm = svm_model.predict(X_test)

print("Accuracy Score is %5.3f" %(accuracy_score(y_test, y_predict_svm)*100))
print(classification_report(y_test, y_predict_svm))

y_predict_proba_svm = svm_model.predict_proba(X_test)[::,1]
fpr_svm, tpr_svm,_=metrics.roc_curve(y_test, y_predict_proba_svm)
auc = round(metrics.roc_auc_score(y_test, y_predict_proba_svm)*100,2)
plt.plot(fpr_svm,tpr_svm,label = "auc ="+str(auc))
plt.title("ROC for SVM Model")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc = 4)
plt.show()

cm_svm = metrics.confusion_matrix(y_test, y_predict_svm, labels = [1,0])
df_cm_svm = pd.DataFrame(cm_svm, index = [i for i in ["1", "0"]], columns = [i for i in ["Predict 1", "Predict 0"]])
sns.heatmap(df_cm_svm, annot = True, fmt = 'g', cmap = "Greens")

#Recall is 83%, Precision is 90% and F1 score is 86%. This model has the best scores among all others
#Accuracy is 97.68%
#Recall score has increased in a huge way and let us see if we can get a better value through Grid Search


# ### Grid Search for SVM
# 
# Note: Running SVM Grid Search on 8 GB RAM computer is a task. It takes almost 3 minutes to get the result. 
# Please be patient, thank you !! 
# 
# I have not given all the parameters to Grid Search due to the limitations of my laptop's RAM capacity.

# In[63]:


grid_param_svm = {"C": [0.1,1,10], "gamma": [0.1,1,10], "kernel": ['rbf', 'linear']}
svm_model = svm.SVC(probability = True)
grid_svm = GridSearchCV(svm_model, grid_param_svm, cv = 10)
grid_svm.fit(X_train, y_train)
y_predict_svmgrid = grid_svm.predict(X_test)


# In[64]:


print("Accuracy Score is %5.3f" %(accuracy_score(y_test, y_predict_svmgrid) * 100))
print(classification_report(y_test, y_predict_svmgrid))
print("Best Score is %5.3f" %(grid_svm.best_score_))
print("Best Parameters for SVM Grid Search is %s" %(grid_svm.best_params_))

#Recall is 83%, Precision is 90% and F1 score is 86%. 
#The values are same as normal SVM because the best params are the default values. 


# ### Evaluate the model. Use confusion matrix to evaluate class level metrics i.e.. Precision and recall. Also reflect the overall score of the model.
# 
# Number of customer responded to the not responded is very less. So, accuracy is not the right measure for the model evaluation.
# 
# ### H0 - Null Hypothesis is that Customer has not responded to a personal loan. 
# ### H1 - Alternate Hypothesis is that Customer has responded to a personal loan.
# 
# Assume a scenario, where the Customer has not bought a Personal Loan but our model has predicted that he would buy a Personal Loan. This is a False Positive case and Bank should be fine with it, since it is not a loss to them. 
# 
# But, consider a scenario where the Customer has bought the Personal Loan, but the model has predicted that he will not buy the Personal Loan. This is a False Negative case and Bank will be at a disadvantage to miss these potential cases. 
# 
# Hence False Negative, a Type 1 error where we reject a null Hypothesis should be a measure for model evaluation. 
# 
# Recall is TP/(TP+FN). If we find a model where the FN is less, then Recall value will be high and we should ideally be choosing a model whose Recall value is the best. 
# 
# In the confusion matrix, we should see the Recall value for 1 (True Positives). 
# 
# ### Logistic Regression - Precision is 82%, Recall is 59% and F1 score is 69%. Accuracy Score is 95.303
# ### KNN Classifier - Precision is 95%, Recall is 66% and F1 score is 78%. Accuracy Score is 96.732
# ### Naive Bayes Classifier - Precision is 42%, Recall is 59% and F1 score is 50%.Accuracy Score is 89.449
# ### SVM Classifier - Precision is 90%, Recall is 83% and F1 Score is 86%. Accuracy Score is 97.686
# 
# 
# ## SVM seems to be the best model for this dataset 
# 

# ### Print classification Report for precision, recall & f-1Score for all the models and evaluate them with proper insights

# 1. Precision Score is 82%, Recall is 59% and F1 Score (Harmonic Mean of Precision and Recall) is 69% for True Positives
# 2. Logistic Regression model has Recall of 59% to identify the True Positives. 

# In[65]:


print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr_grid))


# 1.Precision Score is 95%, Recall is 66% and F1 Score (Harmonic Mean of Precision and Recall) is 78% for True Positives
# 2. KNN Classifier model has Recall of 66% to identify the True Positives. 
# 3. KNN is a better model than Logistic Regression, since Precision and Recall score is better, also F1 score

# In[66]:


print("KNN Classification Report")
print(classification_report(y_test, y_predict_knn_grid))


# Precision Score is 42%, Recall is 59% and F1 Score (Harmonic Mean of Precision and Recall) is 50% for True Positives
# 1. KNN Classifier model has Recall of 59% to identify the True Positives. 
# 2. Though the Recall is better than Logistic Regression and KNN, the Precision takes a hit to a drastic level of 42%.
# 3. F1 score is lesser than Logistic Regression and KNN

# In[67]:


print("Naive Bayes Classification Report")
print(classification_report(y_test, y_predict_nb))


# Precision Score is 90%, Recall is 83% and F1 Score(Harmonic Mean of Precision and Recall) is 86% for True Positives. 
# SVM Model scores are better than all the other models. 

# In[68]:


print("SVM Classification Report")
print(classification_report(y_test, y_predict_svm))


# ### Roc Curve for Model evaluation and explain the difference among all the models used in your analysis
# 
# Area Under the Curve AUC is the best for Support Vector Machine model with 98.21 score. If the AUC is close to 50% then it is equally good for a random guess to work and model does not give a better result. AUC score has to be as close to 100%
# 
# ### Linear Regression AUC score is 94.95
# ### KNN Classifier AUC score is 82.66
# ### Naive Bayes Classifier AUC score is 92.42
# ### Support Vector Machine Classifier AUC score is 98.21
# 
# The difference of the AUC scores is because for each FPR, the TPR is less. The best score will have a higher TPR for every FPR point. 

# Area Under the Curve AUC for Logistic Regression is 94.95

# In[69]:


y_pred_proba_grid = model_grid.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test, y_pred_proba_grid)
auc = round(metrics.roc_auc_score(y_test,y_pred_proba_grid)*100,2)
plt.plot(fpr,tpr, label = "auc="+str(auc))
plt.legend(loc = 4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC of Logistic Regression")
plt.show()


# Area Under the Curve AUC for KNN Classifier is 82.86. 
# This is lesser than Logistic Regression Model

# In[70]:


y_predict_proba_knngrid = grid_model_knn.predict_proba(X_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test, y_predict_knn_grid)
auc = round(metrics.roc_auc_score(y_test, y_predict_knn_grid)*100,2)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc = 4)
plt.title("ROC for KNN Classifier")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# Area Under the Curve AUC for Naive Bayes Classifier is 92.42. 
# The score is better than KNN Classifier but lesser than Logistic Regression. 

# In[71]:


y_predict_proba_nb = nb.predict_proba(X_test)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_test, y_predict_proba_nb)
auc = round(metrics.roc_auc_score(y_test, y_predict_proba_nb)*100,2)
plt.plot(fpr,tpr,label = "auc="+str(auc))
plt.title("ROC for Naive Bayes Classifier")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc = 4)
plt.show()


# In[73]:


svm_model = svm.SVC(probability = True, C = 10, gamma = 0.1, kernel = 'rbf')
svm_model.fit(X_train, y_train)
y_predict_svm = svm_model.predict(X_test)
y_predict_proba_svm = svm_model.predict_proba(X_test)[::,1]
fpr_svm, tpr_svm,_=metrics.roc_curve(y_test, y_predict_proba_svm)
auc = round(metrics.roc_auc_score(y_test, y_predict_proba_svm)*100,2)
plt.plot(fpr,tpr,label = "auc ="+str(auc))
plt.title("ROC for SVM Model")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc = 4)
plt.show()


# ### Calculate the fpr and tpr for all thresholds of the classification

# ### Logistic Regression FPR & TPR

# In[ ]:


fpr_lr,tpr_lr,_=metrics.roc_curve(y_test, y_pred_proba_grid)
fig = sns.scatterplot(fpr_lr,tpr_lr)
fig.set(xlabel='FPR', ylabel='TPR')
fig.set(title = "Logistic Regression FPR Vs TPR")


# ### KNN Classifier FPR & TPR

# In[ ]:


fpr_knn, tpr_knn,_ = metrics.roc_curve(y_test, y_predict_knn_grid)
fig = sns.scatterplot(fpr_knn, tpr_knn)
fig.set(xlabel = 'FPR', ylabel = 'TPR')
fig.set(title = "KNN Classifier FPR Vs TPR")


# ### Naive Bayes Classifier FPR & TPR

# In[ ]:


fpr_nb,tpr_nb,_ = metrics.roc_curve(y_test, y_predict_proba_nb)
fig = sns.scatterplot(fpr_nb, tpr_nb)
fig.set(xlabel = 'FPR', ylabel = "TPR")
fig.set(title = "Naive Bayes FPR Vs TPR")


# In[ ]:


fpr_svm, tpr_svm,_=metrics.roc_curve(y_test, y_predict_proba_svm)
fig = sns.scatterplot(fpr_svm, tpr_svm)
fig.set(xlabel = 'FPR', ylabel = "TPR")
fig.set(title = "SVM FPR Vs TPR")


# ### Discuss some of the key hyper parameters available for the selected algorithm. What values did you initialize these parameters to?
# 
# 
# Logistic Regression - LogisticRegression(solver = "liblinear", C = 10, penalty = 'l2')
# 
# Based on the Grid Search results: 
# Solver is liblinear. For small datasets, ‘liblinear’ is a good choice. 
# C - Inverse of regularization strength; must be a positive float. Smaller value signifies stronger regularization
# Penalty - L2 Lasso Regularization. The coefficients of some of the features will be made to zero
# 
# KNN Classifier - KNeighborsClassifier(n_neighbors= 3, metric= 'manhattan', weights = 'distance')
# 
# n_neighbors - 3 neighbors used for query to classify
# metric - Distance metric used is 'Manhattan'. |x1 - x2| + |y1 - y2|
# weights - ‘distance’: weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
# 
# Naive Bayes Classifier - GaussianNB()
# There are no parameters for hyper tuning in Naive Bayes Classifier. 
# 
# Support Vector Machine - svm.SVC(C: 10, gamma: 0.1, kernel: 'rbf')
# C - Penalty value is 10 for a soft margin vector
# Gamma is 0.1 - Influence of one data point on another (Bias)
# kernel is radial basis function - The algorithm starts with a single data point and based on the proximity it classifies the adjacent points
# 

# ### Regularization techniques used for the model.
# 
# 1. Logistic Regression - Have used L2 Regularization Lasso technique. Lasso assigns the co-efficients of certain features which do not contribute to the independent (target) feature to zero 
# 
# 2. KNN Classifier - n_neighbors used were 3, metric is manhattan and weight is distance. Though these are not regularization parameters, these are the best params out of the Grid Search
# 
# 3. Naive Bayes Classifier - No regularization technique and the algorithm is based on conditional probabilities. 
# 
# 4. SVM Classifier - Penalty C is 10, gamma is 0.1 and kernel used is rbf(radial basis function). Though these are not regularization parameters, these are the best params out of the Grid Search 
# 
# 

# ### Range estimate at 95% confidence for the model performance in production.
# 
# 1. error = Incorrect Predictions/ Total Predictions
# 2. Confidence Interval = error +/- const * sqrt( (error * (1 - error)) / n)
# 3. 95% confidence = 1.96 is constant

# In[ ]:


# Logistic Regression
LR_Error = (57+15)/ (1436)
n = 1436
CI_LR = [LR_Error + 1.96 * np.sqrt((LR_Error * (1-LR_Error)) / n), LR_Error - 1.96 * np.sqrt((LR_Error * (1-LR_Error)) / n)]
print("Confidence Interval for Logistic Regression is ") 
print(CI_LR)
print("------------------------------------------------------------")

# KNN Classifier
KNN_Error = 61/ (1436)
n = 1436
CI_KNN = [KNN_Error + 1.96 * np.sqrt((KNN_Error * (1-KNN_Error)) / n), KNN_Error - 1.96 * np.sqrt((KNN_Error * (1-KNN_Error)) / n)]
print("Confidence Interval for KNN Classifier is ")
print(CI_KNN)
print("------------------------------------------------------------")

# Naive Bayes Classifier
NB_Error = 132/ (1436)
n = 1436
CI_NB = [NB_Error + 1.96 * np.sqrt((NB_Error * (1-NB_Error)) / n), NB_Error - 1.96 * np.sqrt((NB_Error * (1-NB_Error)) / n)]
print("Confidence Interval for NB Classifier is ")
print(CI_NB)
print("------------------------------------------------------------")

#SVM Classifier

SVM_Error = 44/ (1436)
n = 1436
CI_SVM = [SVM_Error + 1.96 * np.sqrt((SVM_Error * (1-SVM_Error)) / n), SVM_Error - 1.96 * np.sqrt((SVM_Error * (1-SVM_Error)) / n)]
print("Confidence Interval for SVM Classifier is ")
print(CI_SVM)
print("------------------------------------------------------------")


# In[ ]:




