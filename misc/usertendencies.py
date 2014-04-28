import matplotlib.pyplot as plt
from sqlalchemy import *
import numpy as np
from sqlalchemy.orm import sessionmaker
from churndata import *
import pandas as pd
from pandas import DataFrame
from util import query_to_df
from rolling_mean import rolling_mean
from util import campaign_to_num,event_to_num,transform_column,hist_and_show,vectorize
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
user_df = user_df.groupby(['Users_id','visit_date'])
user_logins = {}

def add_to_logins(group):
    group = group.apply(pd.to_datetime)
    print group.as_matrix()
    return rolling_mean(group.as_matrix(),window='30d')
user_df = user_df.aggregate(add_to_logins)

print user_df



