import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

###### DATA PLOTTING ######

def plot_data(df):
    '''Plot data from csv using historgrams, boxplots, and correlation heatmapping'''
    sensor_columns = [col for col in df.columns if 'sensor_' in col]
    
    # Histogram
    df[sensor_columns].hist(bins=30, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.suptitle('Sensor Data Distributions')
    plt.show()

    # Box Plot
    plt.figure(figsize=(20, 10))
    df[sensor_columns].boxplot()
    plt.xticks(rotation=90)
    plt.title('Sensor Data Boxplot')
    plt.show()

    # Correlation heatmap between sensors and machine status encoded
    correlation_data = df[sensor_columns + ['machine_status_encoded']]
    correlation_matrix = correlation_data.corr()
    plt.figure(figsize=(24, 20))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", annot_kws={"size": 6})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Correlation Matrix')
    plt.show()


###### EXPLORATORY DATA ANALYSIS ######

def temporal_analysis(df):
    '''Encode timestamp for visualization, then plot time series for each sensor'''
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Select sensor columns
    sensor_columns = [col for col in df.columns if 'sensor_' in col]

    # Plot time series for each sensor
    for sensor in sensor_columns:
        plt.figure(figsize=(15, 7))
        plt.plot(df[sensor], label=sensor)
        plt.title(f"{sensor} Readings Over Time")
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Reading')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def boxplot_sensor_status(df):
    ''' Create boxplot for each sensor '''
    # Convert 'machine_status' to a categorical type
    if not pd.api.types.is_categorical_dtype(df['machine_status']):
        df['machine_status'] = df['machine_status'].astype('category')

    # Identify sensor columns
    sensor_columns = [col for col in df.columns if 'sensor_' in col]
    
    # Plot a boxplot for each sensor
    for sensor in sensor_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='machine_status', y=sensor, data=df)
        plt.title(f'Boxplot of {sensor} by Machine Status')
        plt.ylabel('Sensor Reading')
        plt.xlabel('Machine Status')
        plt.xticks(rotation=45)
        plt.show()
