import numpy as np
import pandas as pd




# Prepare the customer/order data for model
def order_wrangle(filepath):
    order_list = pd.read_excel(filepath)  # Dataframe of orders
    depot = 'A123'  # depot
    order_destinations = order_list['Destination Code'].unique().tolist()
    order_list['Destination Code'] = order_list['Destination Code'].astype(str)  # Ensure data type
    location_codes = [str(depot)] + [str(code) for code in order_destinations]
    return order_list, location_codes


# Prepare the location data for model
def location_wrangle(filepath, location_codes):
    df = pd.read_csv(filepath)
    locations_df = df[df['location_code'].isin(location_codes)]  # Filter the data
    location_codes = locations_df['location_code'].tolist()  # indices are changed

    # Prepare time windows
    time_windows = {location_code: 0 for location_code in location_codes}
    time_window = [
        (int(start.split(':')[0]), int(end.split(':')[0]))
        for start, end in zip(locations_df['location_loading_unloading_window_start'],
                              locations_df['location_loading_unloading_window_end'])
    ]

    for idx, value in enumerate(time_window):
        time_windows[location_codes[idx]] = value

    return locations_df, time_windows


# Prepare the distance and time matrix for model
def matrices_wrangle(filepath, location_codes):
    travel_matrix = pd.read_csv(filepath)  # read travel matric data
    distance_matrix = np.zeros((len(location_codes), len(location_codes)))  # Initialize matrices
    time_matrix = np.zeros((len(location_codes), len(location_codes)))

    # Map location codes to indices
    location_index = {location_codes[i]: i for i in range(len(location_codes))}

    # Create the distance and time matrix
    for _, row in travel_matrix.iterrows():
        if row['source_location_code'] in location_codes and row['destination_location_code'] in location_codes:
            i = location_index[row['source_location_code']]
            j = location_index[row['destination_location_code']]
            distance_matrix[i][j] = row['travel_distance_in_km']
            time_matrix[i][j] = row['travel_time_in_min']

    return distance_matrix, time_matrix, location_index


# Prepare the demand data
def demand_wrangle(location_codes, order_list):
    # Initialize the demand dictionary with zero demand for all locations
    demand_dict = {location_code: 0 for location_code in location_codes}

    # Update the demand dict.
    for _, row in order_list.iterrows():
        destination_code = row['Destination Code']
        demand_dict[destination_code] += row['Total Weight']

    return demand_dict


def capacity_wrangle(filepath):
    trucks = pd.read_csv(filepath)
    # Initialize the capacity dictionary with zero capacity for all trucks
    truck_capacities = {t_id: 0 for t_id in trucks['truck_id']}

    # Update the dict
    for _, row in trucks.iterrows():
        t_id = row['truck_id']
        truck_capacities[t_id] += row['truck_max_weight']

    return truck_capacities






