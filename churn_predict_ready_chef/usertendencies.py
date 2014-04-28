from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from pandas.tseries.index import DatetimeIndex
from pandas import Series
from churn_predict_ready_chef.rolling_mean import rolling_mean
from churndata import *
from util import query_to_df

db = create_engine('sqlite:///forjar.db')


metadata = MetaData(db)

Session = sessionmaker(bind=db)


session = Session()


"""
Counts the users by campaign id
"""
user_dist = session.query(Users).join(Visit,Users.id == Visit.user_id).add_entity(Visit)
user_df = query_to_df(session,user_dist)
print user_df.columns
resamp = user_df.set_index('visit_date').groupby(['Users_id']).resample('M', how='sum')
print resamp
