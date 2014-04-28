from pandas import DataFrame
from rolling_mean import  rolling_mean


def last_k_decrease(dates):
    print rolling_mean(dates,'24h')