## Cleaning the data.

### Reading in the data.

Checking the datatypes, the weather data has missing values, marked by "M".
Altering the read_csv to account for this fixes this issue.

### Setting Date as an index.

### Missing values

The NOAA data for our period of study has plenty of missing values - all of the Water1 column and half of three other columns (Depart, Snowfall, and Depth). None of these, other than Water1, might be related to weather data we want to know for West Nile Virus. But if Water1 is missing from all of the data, we can not use it.

We elected to drop these values, and then for the remaining data, drop rows where there were missing values.

### Setting a hot/dry indicator variable on the weather data

This is something we think could help the model.

### Weather data - averaging the columns by date

Since the weather data has data for two Chicago NOAA stations for each day, these could be averaged by the day.

The function we would use for this is resample. However, even after cleaning the data as best we could, resampling produced null values:

**Dataset Info after cleaning of null values:**
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 2932 entries, 2007-05-01 to 2014-10-31
Data columns (total 16 columns):
Station        2932 non-null int64
Tmax           2932 non-null int64
Tmin           2932 non-null int64
Tavg           2932 non-null float64
DewPoint       2932 non-null int64
WetBulb        2932 non-null float64
Heat           2932 non-null float64
Cool           2932 non-null float64
Sunrise        2932 non-null float64
Sunset         2932 non-null float64
PrecipTotal    2932 non-null object
StnPressure    2932 non-null float64
SeaLevel       2932 non-null float64
ResultSpeed    2932 non-null float64
ResultDir      2932 non-null int64
AvgSpeed       2932 non-null float64
dtypes: float64(10), int64(5), object(1)
memory usage: 389.4+ KB

**DataSet Info after Resample on the cleaned data**
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 2741 entries, 2007-05-01 to 2014-10-31
Freq: D
Data columns (total 15 columns):
Station        1471 non-null float64
Tmax           1471 non-null float64
Tmin           1471 non-null float64
Tavg           1471 non-null float64
DewPoint       1471 non-null float64
WetBulb        1471 non-null float64
Heat           1471 non-null float64
Cool           1471 non-null float64
Sunrise        1471 non-null float64
Sunset         1471 non-null float64
StnPressure    1471 non-null float64
SeaLevel       1471 non-null float64
ResultSpeed    1471 non-null float64
ResultDir      1471 non-null float64
AvgSpeed       1471 non-null float64
dtypes: float64(15)
memory usage: 342.6 KB

This does not make sense to me.

However, cleaning the data enough, and then making a pivot table with date as the index does work (assuming you do not set the index as Date initially.)

**Some patterns in the missing weather data we noticed**

1) Sunrise and Sunset are never collected at Station 2 (but given they're both in the same city, or close enough to each other, this is likely unimportant.)

2) For the most part, the average temperature for a station for a day was the simple average of its high and low. So for missing average temperature data points, taking the simple average of that station's high and low for that day was the imputation method.

3) Similarly, other values (Heat/Cool) and more looked the same or similar to the other station's value for that day. Usually, for each day, if one station was missing that data, the other had it. Creating a pivot table took care of these values, actually, since the mean of one value and one null is that one value for pandas and numpy.

### Spray data - location....
The spraying only occurred for certain days and locations. Matching ideally would be on distance from the spray location and a trap to associate that a location was sprayed on a particular day.

### Joining the tables.

Absent proper location matching for spray data and test sites, and averaging the station data, the merged data is large.


### Initial model.

Classification:
1) Logistic Regression
2) KNN (could also be useful for saying which neighbors are near each other on
everything other than spray)
3) Random Forest Classifier
4) RF Extra Trees
5) DT Classifier AdaBoost

Metrics to judge by:

-Classification Report:
  -Recall and Precision for West Nile Virus being predicted as 1 (Yes)

-AUC.
