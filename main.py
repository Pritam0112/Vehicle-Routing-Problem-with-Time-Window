import pandas as pd 
from pulp import *
import csv
import sys


# orders = []
loc_veh_class = {}  # {'location_code':[vehicle_class]}
with open("data/locations.csv",'r') as f:
    # trucks = f.readlines()
    reader = csv.DictReader(f)
    headers = [header.strip() for header in reader.fieldnames]
    for row in reader:
        # orders.append(row[headers[0]])
        loc_veh_class.update({row[headers[0]]:eval(row[headers[1]])})


vehicles_dict ={}  # {'id' : ['class','max_wt']}
vehicles = []        # list of unique veh_id
vehicle_class = {}   #{'class' : [veh_id]}
 
with open("data/trucks.csv",'r') as f:
    # trucks = f.readlines()
    reader = csv.DictReader(f)
    headers = [header.strip() for header in reader.fieldnames]
    for row in reader:
        vehicles_dict.update({row[headers[3]]:[row[headers[0]],int(row[headers[1]])]})
        vehicles.append(row[headers[3]])
        if row[headers[0]] not in vehicle_class:
            vehicle_class[row[headers[0]]] = [row[headers[3]]]
        else:
            vehicle_class[row[headers[0]]].append(row[headers[3]])

# print('Vehicles available:',len(vehicles))
# print(vehicles_dict)
# sys.exit()

demands = {}     #{order_id:location_code,demand_weight}
orders = []     #list of invoice_ordeer
with open("data/order_list1.csv",'r') as f:
    # trucks = f.readlines()
    reader = csv.DictReader(f)
    headers = [header.strip() for header in reader.fieldnames]
    for row in reader:
        orders.append(row[headers[0]])
        demands.update({row[headers[0]]:[row[headers[2]],float(row[headers[3]])]})

orders.insert(0,'source')
orders.append('sink')
# print(len(orders))
# print(demands)
# sys.exit()
# print('Orders Considered:',orders[:5])

def return_source_dest_code(invoice):
    return demands[invoice][0]

loc_vehicle_id = {}   #{'loc_code' : [veh_id]}

for id_key, vehicle_list in loc_veh_class.items():
    all_vehicle_ids = []
    for vehicle_type in vehicle_list:
        all_vehicle_ids.extend(vehicle_class.get(vehicle_type, []))
    
    loc_vehicle_id[id_key] = all_vehicle_ids

order_vehicle_id = {}    #{'invoice' : [veh_id]}
for ord in orders:
    if (ord == 'source') or (ord == 'sink'):
        order_vehicle_id[ord] = loc_vehicle_id['A123']
    else:
        order_vehicle_id[ord] = loc_vehicle_id[return_source_dest_code(ord)]


# print(order_vehicle_id)
# sys.exit()

class DistanceTravelTime():
    def __init__(self,source_location_code,destination_location_code,travel_distance_in_km,travel_time_in_min):
        self.source_location_code = source_location_code
        self.destination_location_code = destination_location_code
        self.travel_distance_in_km = float(travel_distance_in_km)
        self.travel_time_in_min = float(travel_time_in_min)

def read_travel_matrix(file_path):
    travel = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        headers = [header.strip() for header in reader.fieldnames]
        for row in reader:
            dist = DistanceTravelTime(
                source_location_code=row[headers[0]],
                destination_location_code=row[headers[1]],
                travel_distance_in_km=row[headers[2]],
                travel_time_in_min =row[headers[3]]
            )
            travel.append(dist)
    return travel

travel_matrix = read_travel_matrix("data/travel_matrix.csv")     # List of travel_matrix objects


def return_dist_time(source_code,dest_code):  # returns (dist,time)
    if source_code == 'source':
        source_code = 'A123'
    else:
        source_code = return_source_dest_code(source_code)
    if dest_code == 'sink':
        dest_code = 'A123'
    else:
        dest_code = return_source_dest_code(dest_code)

    for obj in travel_matrix:
        if (obj.source_location_code == source_code ) and (obj.destination_location_code == dest_code):
            return [float(obj.travel_distance_in_km),float(obj.travel_time_in_min)]    
    return [0,0]
# print(return_dist_time('INV_14062024_01','sink')[0])
# print(orders[1:])
# sys.exit()



def build_model(orders,vehicles,demands,vehicles_dict,order_vehicle_id):
    prob = LpProblem("CVRPTW", LpMinimize)
    # ****************************************
    # Defining decision variables
    # ****************************************
    x = {} #Binary x_i,j,v := 1 if vehicle v visits city j after city i; otherwise 0
    for i in orders[:-1]:
        for j in orders[1:]:
            if i!=j:
                if i == 'source' and j == 'sink':
                    continue
                for v in vehicles:
                    if (v in order_vehicle_id[i]) and (v in order_vehicle_id[j]):
                        x[(i,j,v)] = LpVariable('x_' + str(i) + '_' + str(j) + '_' + str(v),cat = 'Binary')
                    
    # for i in orders[:-1]:  #i!='A123'
    #     for j in orders[1:]:   #j!='source'
    #         for v in vehicles:
    #             if i != j :
    #                 x[(i,j,v)] = LpVariable('x_' + str(i) + '_' + str(j) + '_' + str(v),cat = 'Binary')
    
    print('xijv variables',len(x))
    # sys.exit()
    s = {} #Continuous s_i,v : = time vehicle v starts to service customer i
    for i in orders:
        for v in vehicles:
            if i == 'source':      #Assuming loading happens before 8 and vehicle ready to serve from 8
                s[(i,v)] = LpVariable('x_' + str(i) + '_' + str(v),lowBound=480,upBound = 1080, cat = 'Continuous')
            elif i == 'sink':
                s[(i,v)] = LpVariable('x_' + str(i) + '_' + str(v),lowBound=480, cat = 'Continuous')
            else:
                s[(i,v)] = LpVariable('x_' + str(i) + '_' + str(v),lowBound=480,upBound = 1320, cat = 'Continuous')

    print('siv variables',len(s))

    I = {}  # I_v := 1 if vehicle v is used; otherwise 0
    for v in vehicles:
        I[v] = LpVariable('I_' + str(v) , cat = 'Binary')
    # ********************************************
    # Objective
    # ********************************************
    # Minimize total operational cost
    obj_val = 0
    for v in vehicles:
        for i in orders[:-1]:
            for j in orders[1:]:
                if (i,j,v) in x:
                    obj_val += (return_dist_time(i,j)[0])*x[(i,j,v)]

    prob += obj_val

    print("Finished modelling objective")
    # ********************************************
    # Constraints
    # ********************************************
    # Start from depot
    for v in vehicles:
        prob += lpSum(x[('source',j,v)] for j in orders[1:-1] if ('source',j,v) in x) == I[v], f"Source[{('A123',j,v)}]"

    print("Finished modelling Start from depot")

    # End at depot
    for v in vehicles:
        prob += lpSum(x[(i,'sink',v)] for i in orders[1:-1] if (i,'sink',v) in x) == I[v], f"Sink[{(i,'A123',v)}]"

    print("Finished modelling End at depot")

    # Flow Balancing
    for v in vehicles:
        for h in orders[1:-1]:
            prob += lpSum(x[(i,h,v)] for i in orders[:-1] if (i,h,v) in x) == lpSum(x[(h,j,v)] for j in orders[1:] if (h,j,v) in x)

    print("Finished modelling Flow Balancing")

    # Each customer is visited exactly once
    for i in orders[1:-1]:
        aux_sum=0
        for v in vehicles:
                aux_sum += lpSum(x[(i,j,v)] for j in orders[1:-1] if (i,j,v) in x) 
        prob += aux_sum ==1
    print("Finished modelling Each customer is visited exactly once")


    # Vehicle capacity constraint    
    for v in vehicles:
        aux_sum = 0
        for j in orders:
            aux_sum += lpSum([demands[i][1]*x[(i,j,v)] for i in orders[1:-1] if (i,j,v) in x]) 
        prob += aux_sum <= int(vehicles_dict[v][1])
    print("Finished modelling Vehicle capacity constraint")

    # Time window constraints
    #considering time in minutes a_i = 08:00 = 480 mins and b_i = 22:00 = 1320 mins
    for v in vehicles:        #wait_time ignored
        for i in orders[:-1]:
            for j in orders[1:]:
                if i!=j and (i,j,v) in x:
                    prob += s[(i,v)] + 20 + return_dist_time(i,j)[1] - 1e8*(1- x[(i,j,v)]) <= s[(j,v)]

    print("Finished modelling Time window constraints")

    
    # Linking constraints
    for v in vehicles:
        for i in orders[:-1]:
            for j in orders[1:]:
                if i!=j and (i,j,v) in x:
                    prob += x[(i,j,v)] <= I[(v)]

    print("Finished modelling Linking constraints")

    
    # Vehicle Compatibility
    # for i in orders[:-1]:
    #     for j in orders[1:]:
    #         if i!=j:
    #             for v in vehicles:
    #                 if (v in order_vehicle_id[i]) and (v in order_vehicle_id[j]):
    #                     x[(i,j,v)] <= 1
    #                 else:
    #                     x[(i,j,v)] == 0
    # print("Finished Vehicle Compatibility constraints")
    
    # *********************************
    # Solve the problem
    # *********************************
    solver = 'GUROBI' 
    print('-'*50)
    print('Optimization solver', solver , 'called')
    # prob.writeLP("../output/cvrptw.lp")
    # print(prob)
    # print()
    if solver == 'GUROBI':
        prob.solve(GUROBI())
    else:
        prob.solve()

    # Print the status of the solved LP
    print("Status:", LpStatus[prob.status])
    print("objective=", value(prob.objective))

if __name__ == "__main__":
    build_model(orders,vehicles,demands,vehicles_dict,order_vehicle_id)
