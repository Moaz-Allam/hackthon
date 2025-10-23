# Make sure PyVRP is installed first:
# pip install pyvrp

import math
import random
from pyvrp import Model, Client, VehicleType, Vehicle, Depot, Coordinates

# --- Configuration ---
NUM_ORDERS = 50
NUM_DEPOTS = 2
RADIUS = 15  # km
CLUSTER_FACTOR = 0.7

# --- Step 1: Create depots (warehouses) ---
depots = []
for i in range(NUM_DEPOTS):
    x, y = random.uniform(0, 100), random.uniform(0, 100)
    depots.append(Depot(Coordinates(x, y), f"Depot_{i+1}"))

# --- Step 2: Generate clustered customer locations ---
clients = []
for i in range(NUM_ORDERS):
    # Pick a depot center and cluster around it
    depot = random.choice(depots)
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(0, RADIUS * CLUSTER_FACTOR)
    x = depot.coords.x + dist * math.cos(angle)
    y = depot.coords.y + dist * math.sin(angle)

    # Random demand by item type (simulate light, medium, heavy)
    demand = random.randint(1, 10)
    clients.append(Client(Coordinates(x, y), demand, f"Order_{i+1}"))

# --- Step 3: Define vehicle types ---
light_van = VehicleType(capacity=40, fixed_cost=50, cost_per_km=1.0, name="LightVan")
medium_truck = VehicleType(capacity=80, fixed_cost=80, cost_per_km=1.5, name="MediumTruck")
heavy_truck = VehicleType(capacity=120, fixed_cost=120, cost_per_km=2.0, name="HeavyTruck")

# --- Step 4: Build model ---
model = Model()

# Add depots and clients
for depot in depots:
    model.add_depot(depot)
for client in clients:
    model.add_client(client)

# Add vehicles
for depot in depots:
    for _ in range(6):  # LightVan
        model.add_vehicle(Vehicle(light_van, depot))
    for _ in range(4):  # MediumTruck
        model.add_vehicle(Vehicle(medium_truck, depot))
    for _ in range(2):  # HeavyTruck
        model.add_vehicle(Vehicle(heavy_truck, depot))

# --- Step 5: Set distance function (Euclidean) ---
def euclidean(a: Coordinates, b: Coordinates) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

model.set_distance_matrix_function(euclidean)

# --- Step 6: Solve the VRP ---
result = model.solve(time_limit=10)  # 10-second heuristic search

# --- Step 7: Output results ---
print(f"✅ Configuration Valid - Total routes: {len(result.routes())}")
print(f"Total distance: {result.cost():.2f} km\n")

for i, route in enumerate(result.routes(), 1):
    vehicle = route.vehicle
    print(f"Route {i}: {vehicle.vehicle_type.name} from {vehicle.start_depot.name}")
    stops = " -> ".join(client.name for client in route.clients)
    print(f"Stops: {stops}")
    print(f"Load: {route.load()} / {vehicle.vehicle_type.capacity}")
    print(f"Distance: {route.distance():.2f} km\n")
