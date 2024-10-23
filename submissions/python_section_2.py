import pandas as pd
import numpy as np
from haversine import haversine, Unit

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a distance matrix based on the provided dataset of toll locations.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'id', 'latitude', and 'longitude'.

    Returns:
        pandas.DataFrame: Distance matrix where the entry at [i][j] represents the cumulative distance
                          between ID i and ID j, with diagonal values set to 0 and symmetric distances.
    """
    distance_matrix = pd.DataFrame(index=df['id'], columns=df['id'])

    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                loc1 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
                loc2 = (df.iloc[j]['latitude'], df.iloc[j]['longitude'])
                distance = haversine(loc1, loc2, unit=Unit.METERS)
                distance_matrix.iloc[i, j] = distance
            else:
                distance_matrix.iloc[i, j] = 0  # Distance to itself is 0

    # Make the matrix symmetric
    distance_matrix = distance_matrix + distance_matrix.T - np.diag(distance_matrix.values.diagonal())
    
    return distance_matrix


def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix into a long-format DataFrame.

    Args:
        df (pandas.DataFrame): Distance matrix with indices and columns representing IDs.

    Returns:
        pandas.DataFrame: DataFrame containing 'id_start', 'id_end', and 'distance' columns.
    """
    unrolled_df = df.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]  # Exclude self-distance

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance'.
        reference_id (int): The ID to use as the reference for the comparison.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold.
    """
    # Calculate the average distance for the reference ID
    average_distance = df[df['id_start'] == reference_id]['distance'].mean()
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    # Find IDs within the threshold
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_ids = avg_distances[(avg_distances['distance'] >= lower_bound) & 
                                  (avg_distances['distance'] <= upper_bound)]
    
    return filtered_ids.sort_values(by='distance')


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for toll rates by vehicle type.
    """
    # Define rate coefficients
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient
    
    return df


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame containing distance and vehicle types.

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates and additional day/time information.
    """
    # Create columns for start_day, start_time, end_day, and end_time
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['start_day'] = np.random.choice(days, size=len(df))
    df['end_day'] = df['start_day']  # Assume end day is the same for simplicity
    
    # Assign random times for start_time and end_time
    df['start_time'] = pd.to_datetime(np.random.choice(pd.date_range("00:00:00", "23:59:59", freq='T'), size=len(df))).dt.time
    df['end_time'] = pd.to_datetime(np.random.choice(pd.date_range("00:00:00", "23:59:59", freq='T'), size=len(df))).dt.time
    
    # Apply toll rates based on time of day and day of the week
    for index, row in df.iterrows():
        if row['start_day'] in ['Saturday', 'Sunday']:
            # Weekend rate
            for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                df.at[index, vehicle_type] *= 0.7
        else:
            # Weekday rates
            if row['start_time'] < pd.to_datetime('10:00:00').time():
                discount = 0.8
            elif row['start_time'] < pd.to_datetime('18:00:00').time():
                discount = 1.2
            else:
                discount = 0.8
            
            for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                df.at[index, vehicle_type] *= discount

    return df

