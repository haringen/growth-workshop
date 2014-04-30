from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import pandas as pd

from churn_predict_ready_chef import rolling_mean
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
user_df = user_df.groupby(['Users_id','visit_date'])
user_logins = {}

def add_to_logins(group):
    group = group.apply(pd.to_datetime)
    print group.as_matrix()
    return rolling_mean(group.as_matrix(),window='30d')
user_df = user_df.aggregate(add_to_logins)

print user_df



