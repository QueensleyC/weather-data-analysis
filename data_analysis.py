# Import libraries
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.power import GofChisquarePower
from statsmodels.tsa.arima.model import ARIMA

"""
    TASK 1
"""
df = pd.read_csv("CarSharing.csv",index_col = "id") # Imports data and sets the "id" column as the index of the dataset

# duplicated() return True if a row is duplicated, sum() adds up all the "True" values
df.duplicated().sum() 

# Checks for null values per column
df.isnull().sum() 

# A function that plots the distribution of a column and computes the mean, median and mode of a column
def compute_hist_central_tendency(col):
    plt.hist(df[col], bins = 25)
    plt.ylabel("Frequency")
    plt.xlabel(col)
    plt.title("Distribution of "+ col)
    
    print(col, "mean: ", df[col].mean())
    print(col, "median: ", df[col].median())
    print(col, "mode: ", df[col].mode())
    
# filling missing values with medians of the respective columns
df['temp'].fillna(df['temp'].median(), inplace = True)
df['humidity'].fillna(df['humidity'].median(), inplace = True)
df['windspeed'].fillna(df['windspeed'].median(), inplace = True)
df['temp_feel'].fillna(df['temp_feel'].median(), inplace = True)

# Save cleaned data
df.to_csv("car_sharing_cleaned")


"""
    TASK 2
"""

# Read csv data into a dataframe and inspect data
df = pd.read_csv("car_sharing_cleaned",index_col = "id" )
df.head()
df.info()

"""
 Numerical vs Numverical significance test
"""

# Checks if data is normally distributed
def check_normality(data):
    test_stat_normality, p_value_normality=stats.shapiro(data)
    print("p value:%.4f" % p_value_normality)
    if p_value_normality <0.05:
        print("Reject null hypothesis >> The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis >> The data is normally distributed")

# Checks if two columns have the same variance
def check_variance_homogeneity(group1, group2):
    test_stat_var, p_value_var= stats.levene(group1,group2)
    print("p value:%.4f" % p_value_var)
    if p_value_var <0.05:
        print("Reject null hypothesis >> The variances of the samples are different.")
    else:
        print("Fail to reject null hypothesis >> The variances of the samples are same.")


"""
ASSUMPTION CHECK
H₀: The data is normally distributed.

H₁: The data is not normally distributed.

H₀: The variances of the samples are the same.

H₁: The variances of the samples are different.
"""

check_normality(df["temp"])
check_normality(df["temp_feel"])
check_normality(df["humidity"])
check_normality(df["windspeed"])

check_variance_homogeneity(df["temp"], df["temp_feel"])
check_variance_homogeneity(df["temp"], df["humidity"])
check_variance_homogeneity(df["temp"], df["windspeed"])
check_variance_homogeneity(df["temp_feel"], df["humidity"])
check_variance_homogeneity(df["temp_feel"], df["windspeed"])
check_variance_homogeneity(df["humidity"], df["windspeed"])

# Since the data is not normally distributed and more than that,
# have unequal variances, proceed to a non-parametric test.
# Mann Whitney non-parametric test can be used in this scenario

def mannwhitney_test(col1, col2):
    ttest,pvalue = stats.mannwhitneyu(df[col1],df[col2], alternative="two-sided")
    print("p-value:%.4f" % pvalue)
    if pvalue <0.05:
        print("Reject null hypothesis >> it can be said that there is a statistically significant difference between",
              col1, "and",col2)
    else:
        print("Fail to reject null hypothesis")

mannwhitney_test("temp", "temp_feel")
mannwhitney_test("temp", "humidity")
mannwhitney_test("temp", "windspeed")
mannwhitney_test("temp_feel", "humidity")
mannwhitney_test("temp_feel", "windspeed")
mannwhitney_test("humidity", "windspeed")

"""
    Categorical vs Categorical significance test
    Using Chi-square test of independence
"""

def chi_square_test(col1, col2):
    data = df[[col1, col2]]
    table = sm.stats.Table.from_data(data)
    chi_square_test = table.test_nominal_association()
    print("p value:%.4f" % chi_square_test.pvalue)
    if chi_square_test.pvalue > 0.05:        
        print("Accept null hypothesis >> There's no relationship between ", col1, " and ", col2)
    else:
        print("Reject null hypothesis >> There's a relationship between ", col1, " and ", col2)

# Using the method above to test for significance between columns

def hypothesis_season_and_others(compare_with):
    fall_temp = df[df["season"] == "fall"][compare_with]
    spring_temp = df[df["season"] == "spring"][compare_with]    
    summer_temp = df[df["season"] == "summer"][compare_with]    
    winter_temp = df[df["season"] == "winter"][compare_with] 
    
    result = stats.f_oneway(winter_temp.values, summer_temp.values, spring_temp.values, fall_temp.values)   
    print("p value:%.4f" % result.pvalue)
    
    if result.pvalue > 0.05:        
        print("Accept null hypothesis >> The relationship between season and", compare_with, "is not significant")
    else:
        print("Reject null hypothesis >> The relationship between season and", compare_with, "is significant")
    
def hypothesis_weather_and_others(compare_with):
    fall_temp = df[df["weather"] == "Clear or partly cloudy"][compare_with]
    spring_temp = df[df["weather"] == "Light snow or rain"][compare_with]    
    summer_temp = df[df["weather"] == "Mist"][compare_with]    
    winter_temp = df[df["weather"] == "heavy rain/ice pellets/snow + fog"][compare_with] 
    
    result = stats.f_oneway(winter_temp.values, summer_temp.values, spring_temp.values, fall_temp.values)   
    print("p value:%.4f" % result.pvalue)
    
    if result.pvalue > 0.05:        
        print("Accept null hypothesis >> The relationship between season and", compare_with, "is not significant")
    else:
        print("Reject null hypothesis >> The relationship between season and", compare_with, "is significant")
    

hypothesis_weather_and_others("temp")
hypothesis_weather_and_others("temp_feel")
hypothesis_weather_and_others("humidity")
hypothesis_weather_and_others("windspeed")

df.groupby(["holiday"]).describe()

def hypothesis_holiday_others(compare_with):
    holi_yes_temp = df[df["holiday"] == "Yes"][compare_with]
    holi_no_temp = df[df["holiday"] == "No"][compare_with] 
    
    result = stats.ttest_ind(holi_no_temp, holi_yes_temp, equal_var= False)   
    print("p value:%.4f" % result.pvalue)
    
    if result.pvalue > 0.05:        
        print("Accept null hypothesis >> The relationship between holiday and", compare_with, "is not significant")
    else:
        print("Reject null hypothesis >> The relationship between holiday and", compare_with, "is significant")

hypothesis_holiday_others("temp")
hypothesis_holiday_others("temp_feel")
hypothesis_holiday_others("humidity")
hypothesis_holiday_others("windspeed")



df.groupby(["workingday"]).describe()

def hypothesis_workingday_others(compare_with):
    wd_yes_temp = df[df["workingday"] == "Yes"][compare_with]
    wd_no_temp = df[df["workingday"] == "No"][compare_with] 
    
    result = stats.ttest_ind(wd_no_temp, wd_yes_temp, equal_var= True)   
    print("p value:%.4f" % result.pvalue)
    
    if result.pvalue > 0.05:        
        print("Accept null hypothesis >> The relationship between working day and", compare_with, "is not significant")
    else:
        print("Reject null hypothesis >> The relationship between working day and", compare_with, "is significant")

hypothesis_workingday_others("temp")
hypothesis_workingday_others("temp_feel")
hypothesis_workingday_others("humidity")
hypothesis_workingday_others("windspeed")







"""
    TASK 3
"""
df = pd.read_csv("car_sharing_cleaned",index_col = "timestamp", parse_dates=True)

df.drop("id", axis = 1, inplace = True) # drop id column

df_2017 = df.iloc[:5422]

# Creating a template for the plots
fig, ax = plt.subplots(figsize=(15, 6))

# Time series plot of temperature
df_2017["temp"].plot(xlabel = "time", ylabel = "temperature", title = "Temperature Time series", ax = ax)

# Resampling humidity column to daily data and then plotting 
df_h_resample = df_2017["humidity"].resample("D").mean().fillna(method = "ffill")
df_h_resample.plot(xlabel = "time", ylabel = "humidity", title = "Humidity Time series (Daily)", ax = ax)

# Resampling windspeed column to daily data and then plotting 
df_w_resample = df_2017["windspeed"].resample("D").mean().fillna(method = "ffill")
df_w_resample.plot(xlabel = "time", ylabel = "windspeed", title = "Windspeed Time series (Daily)", ax = ax)

# Resampling demand column to daily data and then plotting 
df_d_resample = df_2017["demand"].resample("D").mean().fillna(method = "ffill")
df_d_resample.plot(xlabel = "time", ylabel = "demand", title = "Demand Time series (Daily)", ax = ax)






"""
    TASK 4
"""
# Read csv file into a dataframe and set the timestamp column as the index. 
# parse_dates makes pandas read in the datetime columns as datatime columns
df = pd.read_csv("car_sharing_cleaned",index_col = "timestamp", parse_dates=True)

df.drop("id", axis = 1, inplace = True) # drop id column

# Resampling df to provide the mean "demand" for each week 
# and using forward fill to impute any missing values
df_resample = df["demand"].resample("W").mean().fillna(method = "ffill")

# Creates a layout for the figure to be drawn
fig, ax = plt.subplots(figsize = (15,6)) 
# ACF plot for the data in df_resample
plot_acf(df_resample, ax = ax)

# Creates a layout for the figure to be drawn
fig, ax = plt.subplots(figsize = (15,6)) 
# PACF plot for the data in df_resample
plot_pacf(df_resample, ax = ax)

# Spliting the data
cutoff_test = int(len(df_resample) * 0.70)

y_train = df_resample.iloc[:cutoff_test]
y_test = df_resample.iloc[cutoff_test:]

# Funtion for testing hyperparameters
def arima(p,q):
    # Build model
    model = ARIMA(y_train, order = (p,0,q)).fit()
    # Compute MAE
    start = len(y_train)
    end = len(y_train)+len(y_test)-1
    y_predict = model.predict(start=start, end = end)
    mae = mean_absolute_error(y_test, y_predict)
    print("Test MAE for", p, "0", q, "is:", mae)

# Significant values gotten from PACF and ACF plot
acf_values = [0,1,2,3,4,5]
pacf_values = [0,1,2,11]

# Training a model with every combination of hyperparameters in the lists above
for i in acf_values:
    for j in pacf_values:
        arima(j, i)
    
# The best model is 2,0,3 since it gives the lowest MAE
model = ARIMA(y_train, order = (2,0,3)).fit()

start = len(y_train) # Starting point to be used in predict funtion
end = len(y_train)+len(y_test)-1 # End point to be used in predict funtion
y_predict = model.predict(start=start, end = end) # using model to predict and storing the result
mae = mean_absolute_error(y_test, y_predict) # Calculating MAE
print("Test MAE:", mae)

# A DataFrame with two columns: "y_test" and "y_predict".
# The first contains the true values for the test set, and the second contains the model's predictions

df_pred_test = pd.DataFrame(
    {"y_test": y_test.values, "y_pred": y_predict.values}, index=y_test.index
)

# Time series plot for the values in the dataframe
fig = px.line(df_pred_test, labels={"value": "demand"},range_y=[0,6], title = "")

fig.show()




"""
    TASK 5
"""

# Read in data and drop timestamp column
df = pd.read_csv("car_sharing_cleaned",index_col = "id" )

df.drop("timestamp", axis = 1, inplace = True)

# Encode categorical variables
df_dummy = pd.get_dummies(df)

# Rename columns and drop unnecessary columns
df_dummy.rename(columns={'holiday_Yes':'holiday', "workingday_Yes": "workingday"}, inplace=True)
df_dummy.drop(["holiday_No", "workingday_No"],axis = 1, inplace=True)

# Split data into features and target column
target = "demand"
y = df_dummy[target]
X = df_dummy.drop("demand", axis = 1)

# Split data into training and test sets. Using 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Building and fitting the model of the training dataset
model = RandomForestRegressor(n_estimators= 200, random_state=42).fit(X_train, y_train)

# Using the model to predict values for y
y_pred_training = model.predict(X_test)

# Calculating the Mean Squared Erroe
mse_training = mean_squared_error(y_test,y_pred_training)

print("Training MSE:", mse_training)

# Calculating the coefficient of determination of the Random Forest model (R^2)
model.score(X, y)

# Initialize neural network model
model = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

y_pred_training = model.predict(X_test)
mae_training = mean_absolute_error(y_test, y_pred_training)
mse_training = mean_squared_error(y_test,y_pred_training)
print("Training MAE:", mae_training)
print("Training MSE:", mse_training)






"""
    TASK 6
"""
df = pd.read_csv("car_sharing_cleaned",index_col = "id" )
df.drop("timestamp", axis = 1, inplace = True)

# Creating binary target column
avg_demand = round(df["demand"].mean(), 6)
df["high_demand"] = (df["demand"] > avg_demand).astype(int)
df["high_demand"].replace(0, 2, inplace = True)

df_dummy = pd.get_dummies(df)

df_dummy.drop(["holiday_No", "workingday_No"],axis = 1, inplace=True)

df_dummy.rename(columns={'holiday_Yes':'holiday', "workingday_Yes": "workingday"}, inplace=True)

# Splitting data
target = "high_demand"
X = df_dummy.drop(columns = target)
y = df_dummy[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


model = LogisticRegression(max_iter= 1000).fit(X_train, y_train)
lr_acc = model.score(X_test, y_test)

print("Test Accuracy:", round(lr_acc, 2))

dt_clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
dt_acc = dt_clf.score(X_test,y_test)
print("Test Accuracy:", round(dt_acc, 2))

gb_clf = GradientBoostingClassifier().fit(X_train, y_train)
gb_acc = gb_clf.score(X_test,y_test)
print("Test Accuracy:", round(gb_acc, 2))




"""
    TASK 7
"""
# Read columns needed from the CSV file
df = pd.read_csv("car_sharing_cleaned",usecols=["id", "temp"])

df.head()

# Split the dataset to get data in 2017
df_2017 = df.iloc[:5422]

# Scalar to scale the values in the dataset
scalar = MinMaxScaler()

# Scale columns
df_2017["id_use"] = scalar.fit_transform(df_2017[["id"]])

df_2017["temp_use"] = scalar.fit_transform(df_2017[["temp"]])

# Assign data to be used to variable X
X = df_2017

# Create a clustering function
def kmeans_cluster(n):
    # Build model
    model = KMeans(n_clusters= n, random_state= 42)
    # Fit model to data
    model.fit(X)
    labels = model.labels_
    # Predict data eith model
    y_kmeans = model.predict(X)
    print("For,", n, "clusters, value count is:\n", pd.DataFrame(y_kmeans).value_counts())
    

# Use function created above to cluster data
kmeans_cluster(2)
kmeans_cluster(3)
kmeans_cluster(4)
kmeans_cluster(12)

# Create a Gaussian clustering function
def gaussian_cluster(n):
    # Build model
    model = GaussianMixture(n_components= n, n_init = 5, random_state= 42)
    # Fit model to data
    model.fit(X)
    
    y_kmeans = model.predict(X)
    # Check the distribution
    print("For,", n, "clusters, value count is:\n", pd.DataFrame(y_kmeans).value_counts())
    
gaussian_cluster(2)
gaussian_cluster(3)
gaussian_cluster(4)
gaussian_cluster(12)