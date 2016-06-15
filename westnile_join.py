import pandas as pd

# Read in our weather csv
weather = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/weather_mean.csv')
# Read in the csv for our training data (.5, 1, and 3 mile spray distances)
train_half = pd.read_csv('/Users/Brian/spray_0.5_clean.csv')
train_one = pd.read_csv('/Users/Brian/spray_1_clean.csv')
train_three = pd.read_csv('/Users/Brian/spray_3_clean.csv')

# Iterate through each DataFrame and merge it with the weather DataFrame
for df,name in [(train_half,'0.5'),(train_one,'1'),(train_three,'3')]:
    # Drop the columns created from writing to csv
    df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
    # Get dummy variables for the Species column
    df = pd.get_dummies(df,columns=['Species'])
    # Merge the train DF with the weather DF on the Date columns
    merged = df.merge(weather,on='Date',how='outer')
    # Drop the rows where we have weather data but no traps were checked
    merged.dropna(axis=0,how='any',inplace=True)
    # Save the DataFrame to a csv
    filename = 'spray_' + name + '_merged.csv'
    merged.to_csv(filename)
