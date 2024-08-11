import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
locations_dff = pd.read_csv(r"data/locations.csv")
order_list_df = pd.read_excel(r"data/order_list.xlsx")
travel_matrix_df = pd.read_csv(r"data/travel_matrix.csv")
trucks_df = pd.read_csv(r"data/trucks.csv")


# Convert loading/unloading windows to minutes with explicit format
locations_dff['start_minutes'] = pd.to_datetime(locations_dff['location_loading_unloading_window_start'], format='%H:%M').dt.hour * 60 + pd.to_datetime(locations_dff['location_loading_unloading_window_start'], format='%H:%M').dt.minute
locations_dff['end_minutes'] = pd.to_datetime(locations_dff['location_loading_unloading_window_end'], format='%H:%M').dt.hour * 60 + pd.to_datetime(locations_dff['location_loading_unloading_window_end'], format='%H:%M').dt.minute

# Extract relevant data
locations = locations_dff['location_code'].tolist()
orders = order_list_df.to_dict(orient='records')
travel_matrix = travel_matrix_df.set_index(['source_location_code', 'destination_location_code']).to_dict(orient='index')
trucks = trucks_df.to_dict(orient='records')

depot = 'A123'  # depot
order_destinations = order_list_df['Destination Code'].tolist()
order_list_df['Destination Code'] = order_list_df['Destination Code'].astype(str)  # Ensure data type
location_codes = [depot] + [str(code) for code in order_destinations]
locations_df =  locations_dff[locations_dff['location_code'].isin(location_codes)]

# Convert 'trucks_allowed' column from string representation to actual list
locations_df['trucks_allowed'] = locations_df['trucks_allowed'].apply(eval)
locations_df['trucks_allowed'] = locations_df['trucks_allowed'].apply(set)
locations_df.set_index('location_code', inplace=True)

locations_dfff = locations_df.to_dict(orient='index')
# Create a mapping from truck type to truck IDs
type_to_ids = {}
for _, row in trucks_df.iterrows():
    truck_type = row['truck_type']
    truck_id = row['truck_id']
    if truck_type not in type_to_ids:
        type_to_ids[truck_type] = []
    type_to_ids[truck_type].append(truck_id)
# Initialize the capacity dictionary with zero capacity for all trucks
truck_capacities = {t_id: 0 for t_id in trucks_df['truck_id']}

# capacity of trucks
for _, row in trucks_df.iterrows():
    t_id = row['truck_id']
    truck_capacities[t_id] += row['truck_max_weight']


num_vehicles = len(truck_capacities)


# Define allowed vehicles
allowed_vehicles = {}
for i in range(len(location_codes)):
    for j in range(len(location_codes)):
        if i!=j:
            loc_i = location_codes[i]
            loc_j = location_codes[j]
            trucks_i = locations_dfff[loc_i]['trucks_allowed']
            trucks_j = locations_dfff[loc_j]['trucks_allowed']
            common_truck_types = trucks_i.intersection(trucks_j)
            common_truck_ids = []
            for truck_type in common_truck_types:
                common_truck_ids.extend(type_to_ids.get(truck_type, []))
            allowed_vehicles[(loc_i, loc_j)] = common_truck_ids

service_time_customer = 20
service_time_depot = 60
loc_df = pd.Series(location_codes)



# Create the model
model = gp.Model("CVRPTW2")

# Create decision variables
x = {}
t = {}
y = {}
for k in truck_capacities:
    y[k] = model.addVar(vtype=GRB.BINARY, name=f'y_{k}')
    for i in range(len(location_codes)):
        for j in range(len(location_codes)):
            if i != j:
                x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')
        t[(i, k)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f't_{i}_{k}')

# Objective function: Minimize total distance and fixed costs
obj = gp.quicksum(
    travel_matrix.get((loc_df.iloc[i], loc_df.iloc[j]), {}).get('travel_distance_in_km', 0) * x[(i, j, k)] * (20000 - int(truck_capacities[k]) / 1000)
    for k in truck_capacities
    for i in range(len(location_codes))
    for j in range(len(location_codes)) if (i, j, k) in x and i != 0 and j != 0
) + gp.quicksum(
    int(truck_capacities[k]) * 2 * y[k]
    for k in truck_capacities
)
model.setObjective(obj, GRB.MINIMIZE)

# Flow balancing constraint
for i in range(1, len(location_codes)):
    model.addConstr(
        gp.quicksum(x[(i, j, k)] for j in range(len(location_codes)) for k in truck_capacities if i != j and (i, j, k) in x) ==
        gp.quicksum(x[(j, i, k)] for j in range(len(location_codes)) for k in truck_capacities if i != j and (j, i, k) in x),
        f"Flow_Balancing_{i}"
    )

# Demand constraint
for k in truck_capacities:
    truck_max_weight = int(truck_capacities[k])  # Maximum weight capacity of truck k
    model.addConstr(
        gp.quicksum(
            int(order_list_df['Total Weight'].iloc[i-1]) *
            gp.quicksum(x[(i, j, k)] for j in range(1, len(location_codes)) if (i, j, k) in x)
            for i in range(1, len(location_codes)) # Skipping index 0 as it's depot
        ) <= truck_max_weight * y[k],
        f"Demand_{k}"
    )

# Each vehicle should leave the depot at least once
for k in truck_capacities:
    model.addConstr(
        gp.quicksum(x[(0, j, k)] for j in range(1, len(location_codes)) if (0, j, k) in x) == y[k],
        f"Leave_Depot_{k}"
    )

# Each vehicle should arrive at the depot at least once
for k in truck_capacities:
    model.addConstr(
        gp.quicksum(x[(i, 0, k)] for i in range(1, len(location_codes)) if (i, 0, k) in x) == y[k],
        f"Return_Depot_{k}"
    )

# Ensure each customer is visited exactly once (excluding depot)
for i in range(1, len(location_codes)):  # Exclude depot index 0
    model.addConstr(
        gp.quicksum(
            x[(i, j, k)] for j in range(1, len(location_codes))  # Exclude depot index 0
            for k in truck_capacities if (i, j, k) in x
        ) == 1,
        f"Visit_Customer_{i}"
    )

# Limit the number of vehicles used
model.addConstr(gp.quicksum(y[k] for k in truck_capacities) <= 4)

# Time window constraints
big_M = 1e5  # A large number to effectively deactivate constraints for unused vehicles

for i in range(len(location_codes)):
    for k in truck_capacities:
        start_window = locations_df.loc[str(loc_df.iloc[i]), 'start_minutes'].item()
        end_window = locations_df.loc[str(loc_df.iloc[i]), 'end_minutes'].item()
        model.addConstr(t[(i, k)] >= start_window - big_M * (1 - y[k]), f"Start_Window_{i}_{k}")
        model.addConstr(t[(i, k)] <= end_window + big_M * (1 - y[k]), f"End_Window_{i}_{k}")

# Service time constraints
service_time_customer = 20  # minutes at customer locations
service_time_depot = 60  # minutes at the depot/warehouse

for k in truck_capacities:
    for i in range(len(location_codes)):
        for j in range(len(location_codes)):
            if i != j and (i, j, k) in x:
                travel_time = travel_matrix.get((loc_df.iloc[i], loc_df.iloc[j]), {}).get('travel_time_in_min', 0)
                service_time = service_time_depot if i == 0 else service_time_customer
                model.addConstr(
                    t[(j, k)] >= t[(i, k)] + service_time + travel_time - big_M * (1 - x[(i, j, k)]),
                    f"Service_Time_{i}_{j}_{k}"
                )

# Linking constraint
for k in truck_capacities:
    for i in range(len(location_codes)):
        for j in range(len(location_codes)):
            if (i, j, k) in x:
                model.addConstr(y[k] >= x[(i, j, k)], f"Linking_{i}_{j}_{k}")

# Optimize the model
model.update()
model.optimize()

if model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    model.computeIIS()
    model.write("infeasible_mode1.ilp")
else:
    print("Optimal solution found.")
