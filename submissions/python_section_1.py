from typing import Dict, List
import pandas as pd
import polyline
import re
from itertools import permutations
from haversine import haversine

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        # Create a sublist from i to i + n
        sublist = []
        for j in range(n):
            if i + j < length:
                sublist.append(lst[i + j])
        # Add reversed sublist to result
        for item in reversed(sublist):
            result.append(item)
    
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Sort the dictionary by length
    return dict(sorted(length_dict.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    flat_dict = {}

    def flatten(sub_dict, parent_key=''):
        for k, v in sub_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            elif isinstance(v, list):
                for index, item in enumerate(v):
                    if isinstance(item, dict):
                        flatten(item, f"{new_key}[{index}]")
                    else:
                        flat_dict[f"{new_key}[{index}]"] = item
            else:
                flat_dict[new_key] = v

    flatten(nested_dict)
    return flat_dict


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    """
    return list(map(list, set(permutations(nums))))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    
    return dates


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    """
    # Decode the polyline string
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = 0.0

    # Calculate distances using Haversine formula
    for i in range(1, len(df)):
        df.at[i, 'distance'] = haversine(df.iloc[i - 1], df.iloc[i])

    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    """
    n = len(matrix)
    
    # Rotate the matrix 90 degrees clockwise
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Create a new matrix to hold the transformed values
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Calculate the sum of row and column indices
            total_sum = sum(rotated[i]) + sum(row[j] for row in rotated) - rotated[i][j]
            final_matrix[i][j] = total_sum
            
    return final_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-1 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.
    """
    df['timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    completeness = {}

    for (id1, id2), group in df.groupby(['id', 'id_2']):
        min_time = group['timestamp'].min()
        max_time = group['timestamp'].max()
        # Check for 24 hours and 7 days
        full_24h = (max_time - min_time) >= pd.Timedelta(hours=24)
        full_7d = (group['timestamp'].dt.date.nunique()) >= 7
        completeness[(id1, id2)] = full_24h and full_7d

    return pd.Series(completeness)
