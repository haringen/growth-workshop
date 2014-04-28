from __future__ import division
from sqlalchemy import *
import numpy as np
from sqlalchemy.orm import sessionmaker
from churndata import *
from datetime import datetime
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sqlalchemy import *
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from churndata import *
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame,Series
from pandas.core.groupby import GroupBy
import pandas as pd
from util import query_to_df
from util import *
import matplotlib.pyplot as plt
from sqlalchemy import *
import numpy as np
from sqlalchemy.orm import sessionmaker
from churndata import *
from datetime import datetime
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler



def transform_column(df,column_name,fn):
    """
    Transforms a column with the given function
    """
    df[column_name] = df[column_name].apply(fn)
 
db = create_engine('sqlite:///forjar.db')
 
def query_to_df(session,query):
    """
    Convert an sql query to a pandas data frame
    """
    result = session.execute(query)
    d = DataFrame(result.fetchall())
    d.columns = result.keys()
    return d
 
metadata = MetaData(db)
 
Session = sessionmaker(bind=db)
 
 
session = Session()
 
campaign_to_num = {
	'TW' : 1,
	'RE' : 2,
    'FB' : 3,
    'PI' : 4
}

event_to_num = {
   'like' : 1,
   'share' : 2,
   'nothing' : 3,
   'bought' : 4
}


 
def vectorize_feature_index(df,label_column):
    feature_names = []
    global X,train_index,test_index,y,y_test
    for feature_name in df.columns.values:
        print feature_name
        if feature_name != label_column:
            if feature_name not in feature_names:
                feature_names.append(feature_name)
    
    X = df[feature_names].index
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    train_index,test_index = train_test_split(df.index)
    X = df[feature_names].as_matrix().astype(np.float)
    y = df[label_column].index
    y_test = y[test_index].astype(float)
 
q = session.query(Users).join(Event).add_entity(Event)
df= query_to_df(session,q)
df = df.drop(['Users_id','Event_id','Event_User_Id','Event_Meal_Id','Users_Created_Date'],1)


def to_epoch(time_input):
    return (time_input - datetime(1970,1,1)).total_seconds()
 
transform_column(df,'Event_Type',event_to_num.get)
transform_column(df,'Users_Campaign_ID',campaign_to_num.get)
transform_column(df,'Users_date',to_epoch)
transform_column(df,'Event_date',to_epoch)
vectorize_feature_index(df,'Event_Type')




def transform_column(df,column_name,fn):
    """
    Transforms a column with the given function
    """
    df[column_name] = df[column_name].apply(fn)

db = create_engine('sqlite:///forjar.db')

def query_to_df(session,query):
    """
    Convert an sql query to a pandas data frame
    """
    result = session.execute(query)
    d = DataFrame(result.fetchall())
    d.columns = result.keys()
    return d

metadata = MetaData(db)

Session = sessionmaker(bind=db)


session = Session()

campaign_to_num = {
	'TW' : 1,
	'RE' : 2,
    'FB' : 3,
    'PI' : 4
}

event_to_num = {
   'like' : 1,
   'share' : 2,
   'nothing' : 3,
   'bought' : 4
}



def vectorize_feature_index(df,label_column):
    feature_names = []
    for feature_name in df.columns.values:
        print feature_name
        if feature_name != label_column:
            if feature_name not in feature_names:
                feature_names.append(feature_name)

    X = df[feature_names].index
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    train_index,test_index = train_test_split(df.index)
    X = df[feature_names].as_matrix().astype(np.float)
    y = df[label_column].index
    y_test = y[test_index].astype(float)
    return X,y,train_index,test_index,y_test

q = session.query(Users).join(Event).add_entity(Event)
df= query_to_df(session,q)


df = df.drop(['Users_id','Event_id','Event_User_Id','Event_Meal_Id','Users_Created_Date'],1)

def to_epoch(time_input):
    return (time_input - datetime(1970,1,1)).total_seconds()

transform_column(df,'Event_Type',event_to_num.get)
transform_column(df,'Users_Campaign_ID',campaign_to_num.get)
transform_column(df,'Users_date',to_epoch)
transform_column(df,'Event_date',to_epoch)
X,y,train_index,test_index,y_test = vectorize_feature_index(df,'Event_Type')

def confusion_rates(cm):

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    N = fp + tn
    P = tp + fn

    tpr = tp / P
    fpr = fp / P
    fnr = fn / N
    tnr = tn / N

    rates = np.array([[tpr, fpr], [fnr, tnr]])

    return rates


def profit_curve(classifiers):
    for clf_class in classifiers:
        name, clf_class = clf_class[0], clf_class[1]
        clf = clf_class()
        fit = clf.fit(X[train_index], y[train_index])
        print len(X[train_index])
        probabilities = np.array([prob[0] for prob in fit.predict_proba(X[test_index])])
        profit = []

        indicies = np.argsort(probabilities)[::1]
        print 'indices',indicies
        for idx in xrange(len(indicies)):
            pred_true = indicies[:idx]
            ctr = np.arange(indicies.shape[0])
            masked_prediction = np.in1d(ctr, pred_true)
            cm = confusion_matrix(y_test.astype(int), masked_prediction.astype(int))

            rates = confusion_rates(cm)
            print ('cf for ',idx,rates)

            profit.append(np.sum(np.multiply(rates,cb)))

        plt.plot((np.arange(len(indicies)) / len(indicies) * 100), profit, label=name)
    plt.legend(loc="lower right")
    plt.title("Profits of classifiers")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.ylim(20)
    plt.show()

# Cost-Benefit Matrix
cb = np.array([[4, -5],
               [0, 0]])

# Define classifiers for comparison
classifiers = [("Random Forest", RF),
               ("Logistic Regression", LR),
               ("Gradient Boosting Classifier", GBC)]

# Plot profit curves
profit_curve(classifiers)