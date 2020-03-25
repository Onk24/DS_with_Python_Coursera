
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[227]:

import numpy as np
import pandas as pd


# In[228]:

train_data = pd.read_csv('train.csv', encoding = 'ISO-8859-1')
test_data = pd.read_csv("test.csv", encoding = 'ISO-8859-1')
address = pd.read_csv("addresses.csv")
lat_lon =  pd.read_csv("latlons.csv")


# In[229]:

address = address.set_index("address").join(lat_lon.set_index("address"))


# In[230]:

train = train_data[train_data["compliance"]<=1]


# In[231]:

train = train.set_index("ticket_id").join(address.set_index("ticket_id"))
train = train.reset_index()


# In[232]:

import datetime


# In[233]:

train["ticket_issued_date"] = train["ticket_issued_date"].replace(np.nan, '', regex=True)
train["hearing_date"] = train["hearing_date"].replace(np.nan, '', regex=True)


# In[234]:

def days_diff(df):
    a = df["hearing_date"]
    b = df["ticket_issued_date"]
    #print(a)
    days = 0
    if((a) and (b)):
        xx = abs(datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S'))
        days = xx.days
    return days


# In[235]:

train["days_gap"] = train[["hearing_date", "ticket_issued_date"]].apply(days_diff, axis = 1)


# In[236]:

train["late_fee_flag"] = train["late_fee"]!=0
train["discount_flag"] = train["discount_amount"]!=0


# In[237]:

train = train[["lat","lon", "disposition","judgment_amount" ,"discount_flag", "late_fee_flag","days_gap", "compliance"]]


# In[238]:

#train = pd.get_dummies(train, columns = ["late_fee_flag", "discount_flag", "disposition"], drop_first = True)
train = pd.get_dummies(train, columns = ["disposition"], drop_first = True)


# In[239]:

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
from sklearn.linear_model import LogisticRegression


# In[240]:

train = train.dropna()


# In[241]:

X_train, X_val, y_train, y_val = train_test_split(train.loc[:,train.columns != "compliance"], train["compliance"])


# In[242]:

lr = LogisticRegression()
lr.fit(X_train, y_train)
roc_auc_score(y_val,lr.predict_proba(X_val)[:,1])


# In[254]:

gbc = GradientBoostingClassifier(learning_rate = 0.1)
gbc.fit(X_train, y_train)
roc_auc_score(y_val, gbc.predict_proba(X_val)[:,1])


# In[253]:

abc = AdaBoostClassifier(learning_rate = 1)
abc.fit(X_train, y_train)
roc_auc_score(y_val, abc.predict_proba(X_val)[:,1])


# In[245]:

from sklearn.neural_network import MLPClassifier


# In[246]:

mlp = MLPClassifier(activation = "relu",hidden_layer_sizes = [10,10], alpha = 1)
mlp.fit(X_train, y_train)


# In[247]:

roc_auc_score(y_val, mlp.predict_proba(X_val)[:,1])


# In[255]:

#We select gbc
gbc.fit(train.loc[:,train.columns != "compliance"], train["compliance"])


# In[418]:

import pandas as pd
import numpy as np

def blight_model():
    
    # Your code here
    test_data = pd.read_csv("test.csv", encoding = 'ISO-8859-1')
    test = test_data.set_index("ticket_id").join(address.set_index("ticket_id"))
    test = test.reset_index()
    test["ticket_issued_date"] = test["ticket_issued_date"].replace(np.nan, '', regex=True)
    test["hearing_date"] = test["hearing_date"].replace(np.nan, '', regex=True)
    test["days_gap"] =  test[["hearing_date", "ticket_issued_date"]].apply(days_diff, axis = 1)
    test["late_fee_flag"] = test["late_fee"]!=0
    test["discount_flag"] = test["discount_amount"]!=0
    test = test[["ticket_id","lat","lon", "disposition","judgment_amount" ,"discount_flag", "late_fee_flag","days_gap"]]
    #dispostion status changes
    test["disposition"] = test["disposition"].replace("Responsible - Compl/Adj by Default","Responsible by Default")
    test["disposition"] = test["disposition"].replace("Responsible (Fine Waived) by Admis","Responsible by Admission")
    test["disposition"] = test["disposition"].replace("Responsible by Dismissal","Responsible by Default")
    test["disposition"] = test["disposition"].replace("Responsible - Compl/Adj by Determi","Responsible by Determination")
    test["lat"] = test["lat"].fillna(test["lat"].mean())
    test["lon"] = test["lon"].fillna(test["lon"].mean())
    test = pd.get_dummies(test, columns=["disposition"],drop_first ="True")
    predictions = gbc.predict_proba(test.iloc[:,1:])
    #output = np.hstack((test["ticket_id"].reshape(-1,1), predictions[:,1].reshape(-1,1)))
    output2 = pd.Series(predictions[:,1], index = test["ticket_id"].astype(int))
    return output2# Your answer here

