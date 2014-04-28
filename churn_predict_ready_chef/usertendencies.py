from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
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
user_ids = user_df['Users_id']
user_df['visit_date'] = user_df['visit_date'].apply(pd.to_datetime)
groups = user_df.groupby(['Users_id'])
for name,group in groups:
    print name
    print len(group['visit_date'].index)