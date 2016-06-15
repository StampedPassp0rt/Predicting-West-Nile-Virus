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
weather = pd.read_csv(path_weather, na_values = 'M')

'''2. Converting date columns with pd.to_datetime, setting the index,
and extracting month and year.'''

df_list = [train, spray, weather]

def convert_date(data):
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format = True)
    data.set_index('Date', inplace = True)
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

convert_date(train)
convert_date(spray)
convert_date(weather)

'''3. Create dummy variables for our mosquito species. The alternative is to do this,
but have the three that seem associated with WNV to be grouped together.'''

mosquito_dummies = pd.get_dummies(train.Species, drop_first = True)
train_dummies = pd.merge(train, mosquito_dummies, left_index = True, right_index = True)


'''3. Missing values in weather.'''

np.sum(weather.isnull())

#Drop the columns with a number of missing values... Besides, snowfall is likely irrelevant.
weather.drop(['Depth', 'Water1', 'SnowFall', 'Depart'], axis = 1, inplace = True)

#Drop the rows with missing values...

weather.dropna(axis = 0, how = 'any', inplace = True)
weather.info()

'''3b. take mean of each column for each day for weather, since two stations...'''

for i in weather.index:
    print weather.ix[i]

'''4. Spray has missing time values, but these are likely irrelevant. Hence, no dropping of missing vals.'''

spray.info()

'''5. Join the tables. The train data set is the outer dataset...
What this means is that for years without spray data, the data for spraying will
be missing...'''


merge_temp = pd.merge(train_dummies, weather, how = 'inner', left_index = True, right_index = True)
whole_df = pd.merge(merge_temp, spray, how = 'left', left_index = True, right_index = True)


whole_df.info()

merge_temp.head()

merge_temp.info()

weather.head()

train.info()

whole_df.info()
