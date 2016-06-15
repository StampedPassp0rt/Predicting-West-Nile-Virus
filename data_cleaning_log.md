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
