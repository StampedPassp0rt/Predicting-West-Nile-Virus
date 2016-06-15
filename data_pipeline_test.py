'''Objective of this file:

Demonstrate the pipeline to take the raw data and transform it into the clean
data for analysis.'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in our dataset

#location to read files locally from Ameet's computer:
path_train = "../west_nile/input/train.csv"
path_weather = "../west_nile/input/weather.csv"
path_spray = '../west_nile/input/spray.csv'
path_sample = '../west_nile/input/sample_submission.csv'
path_test = "../west_nile/input/test.csv"

df = pd.read_csv('/Users/Brian/Downloads/west_nile/input/train.csv')

'''1. Reading in data'''

train = pd.read_csv(path_train)
spray = pd.read_csv(path_spray)
weather = pd.read_csv(path_weather, na_values = ['M', '-', ' '])

'''2. Converting date columns with pd.to_datetime'''

df_list = [train, spray, weather]

def convert_date(data):
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format = True)
#    data.set_index('Date', inplace = True)
#    data['Year'] = data.index.year
#    data['Month'] = data.index.month
#    data['Day'] = data.index.day

convert_date(train)
convert_date(spray)


'''3. Create dummy variables for our mosquito species. The alternative is to do this,
but have the three that seem associated with WNV to be grouped together.'''

mosquito_dummies = pd.get_dummies(train.Species, drop_first = True)
train_dummies = pd.merge(train, mosquito_dummies, how = 'inner', left_on = ['Date'], right_on = ['Date'], left_index = True, right_index = True)
train_dummies.info()


'''4. Missing values in weather. Clean and then create pivot to take the average of each column for each day
I.e. a unique row for each day of the measurement period.'''

np.sum(weather.isnull())

#Drop the columns with a number of missing values... Besides, snowfall is likely irrelevant.
weather.drop(['Water1', 'CodeSum'], axis = 1, inplace = True)

weather.head(10)

#Filling sunrise and sunset missing values. Predictable pattern with Midway not collecting this.
weather['Sunrise'] = weather.Sunrise.fillna(method = 'pad', limit = 1)
weather['Sunset'] = weather.Sunset.fillna(method = 'pad', limit = 1)
weather['Depart'] = weather.Depart.fillna(method = 'pad', limit = 1)
weather['SnowFall'] = weather.SnowFall.fillna(method = 'pad', limit = 1)
weather['Depth'] = weather.Depth.fillna(method = 'pad', limit = 1)
#

#Filling avg T for when missing by avg of range....
weather.Tavg[weather.Tavg.isnull() == True] = weather[['Tmax', 'Tmin']][weather.Tavg.isnull() == True].mean(axis = 1)

#Might want to alter so not set on copy...
#Heat and cool are predictable missing for Midway, and looks similar enough that going to do pad from O'Hare.
#Better way to do this?

#weather['Heat'] = weather.Heat.fillna(method = 'pad', limit = 1)
#weather['Cool'] = weather.Cool.fillna(method = 'pad', limit = 1)

#Fill PrecipTotal with 0 - when looking at the PrecipTotal's for the missing value station,
#the other station had no precipitation that day.
weather.PrecipTotal.fillna(0, limit = 1, inplace = True)

#Inspect the table's descriptive statistics pre-combining stations' data.
pd.pivot_table(weather, index = 'Station', aggfunc = 'describe')

#Converting date to datetime for weather.
weather['Date'] = pd.to_datetime(weather['Date'], infer_datetime_format = True)


#Creating pivot table to have average weather stats for each day from the two stations.
weather_mean = pd.pivot_table(weather, index = 'Date')

weather_mean['Year'] = weather_mean.index.year
weather_mean['Month'] = weather_mean.index.month
weather_mean['Day'] = weather_mean.index.day

weather_mean.head(10)


'''5. Spray has missing time values, but these are likely irrelevant. Hence, no dropping of missing vals.'''

spray.info()

'''6. Join the tables. The train data set is the outer dataset...
What this means is that for years without spray data, the data for spraying will
be missing...'''


merge_temp = pd.merge(train_dummies, weather_mean, how = 'left', left_on = 'Date', right_index = True)
whole_df = pd.merge(merge_temp, spray, how = 'left', left_index = True, right_index = True)

train_dummies.info()
weather_mean.info()

whole_df.info()

merge_temp.head()

merge_temp.info()

weather.head()

train.info()

whole_df.info()

'''7. Setting the index to Date,
and extracting month and year.'''

def set_index(data):
    data.set_index('Date', inplace = True)
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

set_index(merge_temp)

merge_temp.info()
