import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as p
from preprocessing import order_wrangle, location_wrangle, demand_wrangle, matrices_wrangle, capacity_wrangle

# Load clean data
order_list, location_codes = order_wrangle(r"data/order_list.xlsx")
locations_df,time_windows = location_wrangle(r"data/locations.csv", location_codes)
distance_matrix, time_matrix, location_index = matrices_wrangle(r"data/travel_matrix.csv", location_codes)
demand_dict = demand_wrangle(location_codes, order_list)
truck_capacities = capacity_wrangle(r"data/trucks.csv")


# Build model and solve
model = gp.Model("CVRPTW")

# x_ijv ∈{0,1} - vehicle v travels from location i to location j; i,j ∈ location_codes
x = model.addVars(location_codes, location_codes, truck_capacities, vtype=GRB.BINARY, name='xijv')
# t_iv ∈ R+.- Arrival time at location i by vehicle v.
t = model.addVars(location_codes, truck_capacities, vtype=GRB.CONTINUOUS, name='tiv')
# y_iv ∈{0,1} - vehicle v services location i
y = model.addVars(location_codes, truck_capacities, vtype=GRB.BINARY, name='yiv')


# objective function : Minimise Fixed and variable cost
model.setObjective(
    gp.quicksum(truck_capacities[v] * 2 * y[i, v] for i in location_codes for v in truck_capacities) +
    gp.quicksum(int((20000 - truck_capacities.get(v)) / 1000) * distance_matrix[location_index[i], location_index[j]] * x[i, j, v] for i in location_codes for j in location_codes for v in truck_capacities),
    GRB.MINIMIZE
)

# Each customer is visited exactly once by one truck
for i in location_codes:
    if i != 'A123':
        model.addConstr(gp.quicksum(x[i, j, v] for j in location_codes for v in truck_capacities if j != i) == 1)

# Flow conservation constraints
for v in truck_capacities:
    for i in location_codes:
        model.addConstr(gp.quicksum(x[i, j, v] for j in location_codes if j != i) - gp.quicksum(x[j, i, v] for j in location_codes if j != i) == 0)

# Vehicle capacity constraints
for v in truck_capacities:
    model.addConstr(gp.quicksum(demand_dict[i] * y[i, v] for i in location_codes) <= truck_capacities[v])

# Ensure a location is visited if it is assigned
for v in truck_capacities:
    for i in location_codes:
        model.addConstr(gp.quicksum(x[i, j, v] for j in location_codes if j != i) == y[i, v])

# Ensure each trip starts at the depot
for v in truck_capacities:
    model.addConstr(gp.quicksum(x['A123', j, v] for j in location_codes if j != 'A123') == 1)

# Ensure each trip ends at the depot
for v in truck_capacities:
    model.addConstr(gp.quicksum(x[i, 'A123', v] for i in location_codes if i != 'A123') == 1)

# Time window constraints
for v in truck_capacities:
    for i in location_codes:
        if i != 'A123':
            model.addConstr(t[i, v] >= time_windows[i][0])
            model.addConstr(t[i, v] <= time_windows[i][1])
        if i == 'A123':
            model.addConstr(t[i, v] >= 0)
            model.addConstr(t[i, v] <= time_windows[i][1])

# Service time and travel time constraints
for v in truck_capacities:
    for i in location_codes:
        for j in location_codes:
            if i != j:
                model.addConstr(t[j, v] >= t[i, v] + time_matrix[location_index[i], location_index[j]] - (1 - x[i, j, v]) * 1e5)

# Optimize the model
model.optimize()

# Print the solution
if model.status == GRB.OPTIMAL:
    for v in truck_capacities:
        for i in location_codes:
            for j in location_codes:
                if x[i, j, v].x > 0.5:
                    print(f"Truck {v} travels from {i} to {j}")

else:
    print("No optimal solution found")