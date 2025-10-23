#!/usr/bin/env python3
"""
Optimized Multi-Warehouse Vehicle Routing Problem Solver
Focuses on: 100% fulfillment, cost minimization, efficient multi-order routes
Fixed: Vehicle assignment tracking to prevent duplicate route assignments
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import heapq


class OptimizedMWVRPSolver:
    """Optimized solver for Multi-Warehouse VRP"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.adjacency_list = {}
        self.distance_cache = {}
        self.used_vehicles = set()  # Track vehicles already assigned to routes
        
    def dijkstra(self, start_node: int, end_node: int) -> Tuple[Optional[List[int]], Optional[float]]:
        """Find shortest path using Dijkstra's algorithm with caching"""
        cache_key = (start_node, end_node)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        if start_node == end_node:
            return [start_node], 0.0
        
        distances = {start_node: 0.0}
        previous = {start_node: None}
        pq = [(0.0, start_node)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == end_node:
                path = []
                node = end_node
                while node is not None:
                    path.append(node)
                    node = previous[node]
                path.reverse()
                result = (path, distances[end_node])
                self.distance_cache[cache_key] = result
                return result
            
            if current not in self.adjacency_list:
                continue
            
            neighbors = self.adjacency_list[current]
            
            for item in neighbors:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    neighbor, dist = item[0], item[1]
                else:
                    neighbor = item
                    dist = self.env.get_distance(current, neighbor)
                    if dist is None:
                        continue
                
                if neighbor in visited:
                    continue
                    
                new_dist = current_dist + dist
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        self.distance_cache[cache_key] = (None, None)
        return None, None
    
    def get_vehicle_capacity_info(self, vehicle_id: str) -> Tuple[float, float]:
        """Get vehicle max capacity (weight, volume)"""
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        return (vehicle.capacity_weight, vehicle.capacity_volume)
    
    def calculate_order_requirements(self, order_id: str) -> Tuple[float, float]:
        """Calculate total weight and volume needed for an order"""
        requirements = self.env.get_order_requirements(order_id)
        total_weight = 0.0
        total_volume = 0.0
        
        for sku_id, quantity in requirements.items():
            sku = self.env.get_sku_details(sku_id)
            total_weight += sku['weight'] * quantity
            total_volume += sku['volume'] * quantity
        
        return (total_weight, total_volume)
    
    def can_add_order_to_vehicle(self, vehicle_id: str, order_id: str, 
                                 current_weight: float, current_volume: float) -> bool:
        """Check if order can fit in vehicle given current load"""
        max_weight, max_volume = self.get_vehicle_capacity_info(vehicle_id)
        order_weight, order_volume = self.calculate_order_requirements(order_id)
        
        return (current_weight + order_weight <= max_weight and 
                current_volume + order_volume <= max_volume)
    
    def find_best_warehouse_for_sku(self, sku_id: str, quantity: int, 
                                    from_node: int) -> Optional[Tuple[str, int, float]]:
        """Find best warehouse with SKU considering distance and inventory"""
        warehouses_with_sku = self.env.get_warehouses_with_sku(sku_id, quantity)
        
        if not warehouses_with_sku:
            return None
        
        best_warehouse = None
        best_distance = float('inf')
        best_node = None
        
        for wh_id in warehouses_with_sku:
            warehouse = self.env.get_warehouse_by_id(wh_id)
            wh_node = warehouse.location.id
            _, distance = self.dijkstra(from_node, wh_node)
            
            if distance is not None and distance < best_distance:
                best_distance = distance
                best_warehouse = wh_id
                best_node = wh_node
        
        return (best_warehouse, best_node, best_distance) if best_warehouse else None
    
    def cluster_orders_by_proximity(self, order_ids: List[str], max_clusters: int) -> List[List[str]]:
        """Cluster nearby orders together - ensures ALL orders are assigned"""
        if not order_ids:
            return []
        
        clusters = [[] for _ in range(max_clusters)]
        order_locations = {oid: self.env.get_order_location(oid) for oid in order_ids}
        
        # Round-robin assignment
        cluster_idx = 0
        for order_id in order_ids:
            clusters[cluster_idx].append(order_id)
            cluster_idx = (cluster_idx + 1) % max_clusters
        
        # Remove empty clusters
        clusters = [c for c in clusters if c]
        
        return clusters
    
    def create_multi_order_route(self, vehicle_id: str, order_ids: List[str]) -> Optional[Dict]:
        """Create route serving multiple orders"""
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        home_node = self.env.get_vehicle_home_warehouse(vehicle_id)
        
        steps = []
        current_node = home_node
        current_weight = 0.0
        current_volume = 0.0
        
        # Start at home
        steps.append({
            'node_id': home_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
        
        # Collect all required items for all orders
        all_requirements = defaultdict(int)
        order_requirements_map = {}
        feasible_orders = []
        
        for order_id in order_ids:
            # Check capacity
            if not self.can_add_order_to_vehicle(vehicle_id, order_id, current_weight, current_volume):
                print(f"  ⚠ Order {order_id} doesn't fit in vehicle {vehicle_id}")
                continue
            
            requirements = self.env.get_order_requirements(order_id)
            order_requirements_map[order_id] = requirements
            feasible_orders.append(order_id)
            
            for sku_id, qty in requirements.items():
                all_requirements[sku_id] += qty
            
            # Update current load
            order_weight, order_volume = self.calculate_order_requirements(order_id)
            current_weight += order_weight
            current_volume += order_volume
        
        if not feasible_orders:
            print(f"  ✗ No feasible orders for vehicle {vehicle_id}")
            return None
        
        print(f"  Processing {len(feasible_orders)} feasible orders")
        
        # Collect items from warehouses (grouped by SKU)
        for sku_id, total_qty in all_requirements.items():
            result = self.find_best_warehouse_for_sku(sku_id, total_qty, current_node)
            
            if not result:
                print(f"  ✗ Cannot find warehouse with {total_qty} of {sku_id}")
                return None
            
            wh_id, wh_node, _ = result
            
            # Navigate to warehouse
            if current_node != wh_node:
                path, _ = self.dijkstra(current_node, wh_node)
                if not path or len(path) < 2:
                    print(f"  ✗ No path to warehouse {wh_id}")
                    return None
                
                # Add intermediate nodes
                for node in path[1:-1]:
                    if not steps or steps[-1]['node_id'] != node:
                        steps.append({
                            'node_id': node,
                            'pickups': [],
                            'deliveries': [],
                            'unloads': []
                        })
                
                current_node = wh_node
            
            # Pickup at warehouse
            if steps and steps[-1]['node_id'] == wh_node:
                steps[-1]['pickups'].append({
                    'warehouse_id': wh_id, 
                    'sku_id': sku_id, 
                    'quantity': total_qty
                })
            else:
                steps.append({
                    'node_id': wh_node,
                    'pickups': [{'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': total_qty}],
                    'deliveries': [],
                    'unloads': []
                })
        
        # Deliver to each order
        delivered_count = 0
        for order_id in feasible_orders:
            order_loc = self.env.get_order_location(order_id)
            
            # Navigate to order
            if current_node != order_loc:
                path, _ = self.dijkstra(current_node, order_loc)
                if not path or len(path) < 2:
                    print(f"  ⚠ No path to order {order_id} at {order_loc}")
                    continue
                
                for node in path[1:-1]:
                    if not steps or steps[-1]['node_id'] != node:
                        steps.append({
                            'node_id': node,
                            'pickups': [],
                            'deliveries': [],
                            'unloads': []
                        })
                
                current_node = order_loc
            
            # Deliver all items for this order
            deliveries = [
                {'order_id': order_id, 'sku_id': sku_id, 'quantity': qty}
                for sku_id, qty in order_requirements_map[order_id].items()
            ]
            
            if steps and steps[-1]['node_id'] == order_loc:
                steps[-1]['deliveries'].extend(deliveries)
            else:
                steps.append({
                    'node_id': order_loc,
                    'pickups': [],
                    'deliveries': deliveries,
                    'unloads': []
                })
            
            delivered_count += 1
        
        print(f"  Deliveries planned: {delivered_count}/{len(feasible_orders)}")
        
        # Return home
        if current_node != home_node:
            path, _ = self.dijkstra(current_node, home_node)
            if path and len(path) >= 2:
                for node in path[1:-1]:
                    if not steps or steps[-1]['node_id'] != node:
                        steps.append({
                            'node_id': node,
                            'pickups': [],
                            'deliveries': [],
                            'unloads': []
                        })
                
                if not steps or steps[-1]['node_id'] != home_node:
                    steps.append({
                        'node_id': home_node,
                        'pickups': [],
                        'deliveries': [],
                        'unloads': []
                    })
        
        if delivered_count == 0:
            print(f"  ✗ No deliveries possible")
            return None
        
        return {
            'vehicle_id': vehicle_id,
            'steps': steps
        }
    
    def get_unused_vehicles(self, available_vehicles: List[str]) -> List[str]:
        """Get list of vehicles that haven't been assigned to routes yet"""
        return [v for v in available_vehicles if v not in self.used_vehicles]
    
    def solve(self) -> Dict:
        """Main solving function"""
        print("=" * 60)
        print("OPTIMIZED MULTI-WAREHOUSE VRP SOLVER")
        print("=" * 60)
        
        # Load network
        print("Loading road network...")
        road_network = self.env.get_road_network_data()
        
        if 'adjacency_list' in road_network:
            self.adjacency_list = road_network['adjacency_list']
        else:
            edges = road_network.get('edges', [])
            for edge in edges:
                from_node = edge.get('from')
                to_node = edge.get('to')
                length = edge.get('length', 1.0)
                
                if from_node and to_node:
                    if from_node not in self.adjacency_list:
                        self.adjacency_list[from_node] = []
                    self.adjacency_list[from_node].append((to_node, length))
        
        print(f"Network loaded: {len(self.adjacency_list)} nodes")
        
        # Get data
        order_ids = self.env.get_all_order_ids()
        available_vehicles = self.env.get_available_vehicles()
        
        print(f"Orders: {len(order_ids)}")
        print(f"Available vehicles: {len(available_vehicles)}")
        
        # Cluster orders for efficient routing
        num_vehicles = len(available_vehicles)
        order_clusters = self.cluster_orders_by_proximity(order_ids, num_vehicles)
        
        print(f"Created {len(order_clusters)} order clusters")
        
        solution = {"routes": []}
        fulfilled_orders = set()
        
        # First pass: Assign clustered orders to vehicles
        for idx, (vehicle_id, order_cluster) in enumerate(zip(available_vehicles, order_clusters)):
            if not order_cluster:
                continue
            
            print(f"\n{'='*50}")
            print(f"Vehicle {idx+1}/{num_vehicles}: {vehicle_id}")
            print(f"  Orders assigned: {order_cluster}")
            
            route = self.create_multi_order_route(vehicle_id, order_cluster)
            
            if route:
                solution['routes'].append(route)
                self.used_vehicles.add(vehicle_id)  # Mark vehicle as used
                
                # Track fulfilled orders
                for step in route['steps']:
                    for delivery in step.get('deliveries', []):
                        fulfilled_orders.add(delivery['order_id'])
                print(f"  ✓ Route created: {len(route['steps'])} steps")
            else:
                print(f"  ✗ Failed to create route")
        
        # Find unfulfilled orders
        all_order_set = set(order_ids)
        unfulfilled_orders = list(all_order_set - fulfilled_orders)
        
        if unfulfilled_orders:
            print(f"\n{'='*60}")
            print(f"SECOND PASS: Handling {len(unfulfilled_orders)} unfulfilled orders")
            print(f"Unfulfilled: {unfulfilled_orders}")
            print('='*60)
            
            # Get unused vehicles
            unused_vehicles = self.get_unused_vehicles(available_vehicles)
            print(f"Unused vehicles available: {len(unused_vehicles)}")
            
            # Try to assign unfulfilled orders to unused vehicles only
            for order_id in unfulfilled_orders:
                if not unused_vehicles:
                    print(f"  ✗ No more unused vehicles for {order_id}")
                    break
                
                # Try each unused vehicle
                assigned = False
                for vehicle_id in unused_vehicles[:]:  # Copy list to allow modification
                    route = self.create_multi_order_route(vehicle_id, [order_id])
                    if route:
                        solution['routes'].append(route)
                        self.used_vehicles.add(vehicle_id)
                        unused_vehicles.remove(vehicle_id)  # Remove from available
                        fulfilled_orders.add(order_id)
                        print(f"  ✓ {order_id} assigned to {vehicle_id}")
                        assigned = True
                        break
                
                if not assigned:
                    print(f"  ✗ Could not assign {order_id} to any unused vehicle")
        
        print()
        print("="*60)
        print(f"✓ Solution generated: {len(solution['routes'])} routes")
        print(f"  Vehicles used: {len(self.used_vehicles)}/{len(available_vehicles)}")
        print(f"  Orders fulfilled: {len(fulfilled_orders)}/{len(order_ids)}")
        print(f"  Fulfillment rate: {len(fulfilled_orders)/len(order_ids)*100:.1f}%")
        
        if len(fulfilled_orders) < len(order_ids):
            remaining = all_order_set - fulfilled_orders
            print(f"  ⚠ Unfulfilled orders: {list(remaining)}")
        
        print("=" * 60)
        
        return solution


def solver(env: LogisticsEnvironment) -> Dict:
    """Main solver entry point"""
    optimizer = OptimizedMWVRPSolver(env)
    return optimizer.solve()


# For testing locally (comment out when submitting)
# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     solution = solver(env)
#     print(f"\nRoutes: {len(solution['routes'])}")