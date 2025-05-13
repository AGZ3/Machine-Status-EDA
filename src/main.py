import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from data_analysis import plot_data, temporal_analysis, boxplot_sensor_status

# load the dataset
df = pd.read_csv('data/sensor.csv')

###### DATA REPROCESSING ######
def reprocess_data(df):
    ''' Format and standarize all data '''

    # Numerical columns: fill missing values with mean
    for column in df.columns:
        if df[column].dtype == float: # used float since all data in the dataset are float format
            df[column].fillna(df[column].mean(), inplace=True)

    # Remove columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # Identify sensor columns
    sensor_columns = [col for col in df.columns if 'sensor_' in col] 
    
    # Loop over each sensor column
    for column in sensor_columns:
        # Calculate Q1 and Q3, then IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Capping the outliers
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)  

    # Standardize sensor readings to have zero mean and one variance
    df[sensor_columns] = df[sensor_columns].apply(stats.zscore, axis=0)

    # Rounding to clean up after all transformations
    df[sensor_columns] = df[sensor_columns].round(2)

    # Encode machine status for data visualization
    df['machine_status_encoded'] = df['machine_status'].astype('category').cat.codes

    # Save the processed DataFrame to a new CSV file
    df.to_csv('data/processed_sensor_data.csv', index=False)
    print("Data processed and saved to 'data/processed_sensor_data.csv'.")

    return df



###### FEATURE ENGINEERING ######

def feature_engineering(df):
    '''Calculate rolling statistics and time-lagged features'''

    # Hand-picked sensors - details in report
    relevant_sensors = ['sensor_00', 'sensor_13', 'sensor_22', 'sensor_23', 'sensor_24', 
                        'sensor_25', 'sensor_26', 'sensor_29', 'sensor_36', 'sensor_48', 'sensor_50']  
    
    # Directly add rolling statistics and lagged features to the existing DataFrame
    for sensor in relevant_sensors:
        df[f'{sensor}_rolling_mean'] = df[sensor].rolling(window=5, min_periods=1).mean()
        df[f'{sensor}_rolling_std'] = df[sensor].rolling(window=5, min_periods=1).std()
        df[f'{sensor}_lag1'] = df[sensor].shift(1)
    
    # Drop rows with null values created as a result of rolling stats and lagged features
    df = df.dropna()

    # Save the engineered features to a new CSV file
    df.to_csv('data/engineered_sensor_data.csv', index=False)

    return df



if __name__=="__main__":


    # reprocess_data(df)

    processed_df = pd.read_csv('data/processed_sensor_data.csv')
    
    # Visualize data trends and patterns
    plot_data(processed_df)

    # Choose a sensor to receive a temporal analysis on
    temporal_analysis(processed_df) 
    # boxplot_sensor_status(processed_df)

    # Engineered Features
    # df_features = feature_engineering(processed_df)

    # engineered_df = pd.read_csv('data/engineered_sensor_data.csv')

