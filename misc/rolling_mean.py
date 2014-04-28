from pandas import Series, DataFrame
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def rolling_mean(data, window, min_periods=1, center=False):
    ''' Function that computes a rolling mean

    Parameters
    ----------
    data : DataFrame or Series
           If a DataFrame is passed, the rolling_mean is computed for all columns.
    window : int or string
             If int is passed, window is the number of observations used for calculating
             the statistic, as defined by the function pd.rolling_mean()
             If a string is passed, it must be a frequency string, e.g. '90S'. This is
             internally converted into a DateOffset object, representing the window size.
    min_periods : int
                  Minimum number of observations in window required to have a value.

    Returns
    -------
    Series or DataFrame, if more than one column
    '''
    def f(x):
        '''Function to apply that actually computes the rolling mean'''
        offset = pd.datetools.to_offset(window)

        if center == False:
            dslice = col[x-offset.delta+timedelta(0,0,1):x]
                # adding a microsecond because when slicing with labels start and endpoint
                # are inclusive
        else:
            dslice = col[x-offset.delta/2+timedelta(0,0,1):
                         x+pd.datetools.to_offset(window).delta/2]
        if dslice.size < min_periods:
            return np.nan
        else:
            return dslice.mean()

    data = DataFrame(data.copy())
    dfout = DataFrame()
    if isinstance(window, int):
        dfout = pd.rolling_mean(data, window, min_periods=min_periods, center=center)
    elif isinstance(window, basestring):
        idx = Series(pd.to_datetime(data.index), index=data.index)
        for colname, col in data.iterkv():
            result = idx.apply(f)
            result.name = colname
            dfout = dfout.join(result, how='outer')
    if dfout.columns.size == 1:
        dfout = dfout.ix[:,0]
    return dfout


# Example
idx = [datetime(2011, 2, 7, 0, 0),
       datetime(2011, 2, 7, 0, 1),
       datetime(2011, 2, 7, 0, 1, 30),
       datetime(2011, 2, 7, 0, 2),
       datetime(2011, 2, 7, 0, 4),
       datetime(2011, 2, 7, 0, 5),
       datetime(2011, 2, 7, 0, 5, 10),
       datetime(2011, 2, 7, 0, 6),
       datetime(2011, 2, 7, 0, 8),
       datetime(2011, 2, 7, 0, 9)]
idx = pd.Index(idx)
vals = np.arange(len(idx)).astype(float)
s = Series(vals, index=idx)
rm = rolling_mean(s, window='24H')

print rm