#!/usr/bin/env python3
"""
Enhanced Multi-Warehouse Vehicle Routing Problem Solver
Focus: 100% fulfillment through aggressive route consolidation
Key improvements:
- Multi-pass assignment (pack more orders per vehicle)
- Greedy bin-packing approach for vehicle utilization
- Reserve vehicles for difficult orders
- Fallback single-order routes only when necessary
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import heapq


class EnhancedMWVRPSolver:
    """Enhanced solver focusing on 100% fulfillment through better consolidation"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.adjacency_list = {}
        self.distance_cache = {}
        self.used_vehicles = set()
        
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
    
    def calculate_order_requirements(self, order_id: str) -> Tuple[float, float, Dict]:
        """Calculate total weight, volume, and SKU requirements for an order"""
        requirements = self.env.get_order_requirements(order_id)
        total_weight = 0.0
        total_volume = 0.0
        
        for sku_id, quantity in requirements.items():
            sku = self.env.get_sku_details(sku_id)
            total_weight += sku['weight'] * quantity
            total_volume += sku['volume'] * quantity
        
        return (total_weight, total_volume, requirements)
    
    def calculate_order_difficulty(self, order_id: str) -> float:
        """Calculate order difficulty score (higher = more difficult)"""
        weight, volume, requirements = self.calculate_order_requirements(order_id)
        order_loc = self.env.get_order_location(order_id)
        
        # Factor 1: Capacity ratio (normalized)
        weight_ratio = weight / 1000.0
        volume_ratio = volume / 10.0
        capacity_score = (weight_ratio + volume_ratio) / 2.0
        
        # Factor 2: SKU diversity
        sku_diversity_score = len(requirements) / 3.0
        
        # Factor 3: Minimum distance to any warehouse
        min_distance = float('inf')
        for wh_id, warehouse in self.env.warehouses.items():
            wh_node = warehouse.location.id
            _, distance = self.dijkstra(order_loc, wh_node)
            if distance is not None and distance < min_distance:
                min_distance = distance
        
        distance_score = min_distance / 50.0 if min_distance != float('inf') else 1.0
        
        difficulty = (0.4 * capacity_score + 0.3 * sku_diversity_score + 0.3 * distance_score)
        return difficulty
    
    def greedy_pack_orders_to_vehicle(self, vehicle_id: str, candidate_orders: List[str]) -> List[str]:
        """
        Greedy bin-packing: pack as many orders as possible into vehicle
        Returns list of orders that fit
        """
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        max_weight = vehicle.capacity_weight
        max_volume = vehicle.capacity_volume
        
        # Sort by size (smallest first for better packing)
        order_sizes = []
        for order_id in candidate_orders:
            weight, volume, _ = self.calculate_order_requirements(order_id)
            size_score = (weight / max_weight + volume / max_volume) / 2.0
            order_sizes.append((size_score, order_id))
        
        order_sizes.sort()  # Smallest first
        
        packed_orders = []
        current_weight = 0.0
        current_volume = 0.0
        
        for _, order_id in order_sizes:
            weight, volume, _ = self.calculate_order_requirements(order_id)
            
            if (current_weight + weight <= max_weight and 
                current_volume + volume <= max_volume):
                packed_orders.append(order_id)
                current_weight += weight
                current_volume += volume
        
        return packed_orders
    
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
    
    def create_multi_order_route(self, vehicle_id: str, order_ids: List[str]) -> Optional[Dict]:
        """Create route serving multiple orders"""
        if not order_ids:
            return None
        
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        home_node = self.env.get_vehicle_home_warehouse(vehicle_id)
        
        steps = []
        current_node = home_node
        
        # Start at home
        steps.append({
            'node_id': home_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
        
        # Collect all required items
        all_requirements = defaultdict(int)
        order_requirements_map = {}
        
        for order_id in order_ids:
            requirements = self.env.get_order_requirements(order_id)
            order_requirements_map[order_id] = requirements
            
            for sku_id, qty in requirements.items():
                all_requirements[sku_id] += qty
        
        # Collect items from warehouses
        for sku_id, total_qty in all_requirements.items():
            result = self.find_best_warehouse_for_sku(sku_id, total_qty, current_node)
            
            if not result:
                return None
            
            wh_id, wh_node, _ = result
            
            # Navigate to warehouse
            if current_node != wh_node:
                path, _ = self.dijkstra(current_node, wh_node)
                if not path or len(path) < 2:
                    return None
                
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
        
        # Deliver to orders (sort by distance for efficient routing)
        order_distances = []
        for order_id in order_ids:
            order_loc = self.env.get_order_location(order_id)
            _, distance = self.dijkstra(current_node, order_loc)
            if distance is not None:
                order_distances.append((distance, order_id))
        
        order_distances.sort()  # Nearest first
        
        delivered_orders = []
        for _, order_id in order_distances:
            order_loc = self.env.get_order_location(order_id)
            
            # Navigate to order
            if current_node != order_loc:
                path, _ = self.dijkstra(current_node, order_loc)
                if not path or len(path) < 2:
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
            
            delivered_orders.append(order_id)
        
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
        
        if not delivered_orders:
            return None
        
        return {
            'vehicle_id': vehicle_id,
            'steps': steps
        }
    
    def solve(self) -> Dict:
        """Main solving function with aggressive consolidation"""
        print("=" * 60)
        print("ENHANCED MWVRP SOLVER - AGGRESSIVE CONSOLIDATION")
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
        all_orders = self.env.get_all_order_ids()
        all_vehicles = self.env.get_available_vehicles()
        
        print(f"Orders: {len(all_orders)}")
        print(f"Available vehicles: {len(all_vehicles)}")
        
        solution = {"routes": []}
        fulfilled_orders = set()
        remaining_orders = set(all_orders)
        remaining_vehicles = all_vehicles.copy()
        
        # Sort vehicles by capacity (largest first for consolidation)
        vehicle_capacities = []
        for v_id in all_vehicles:
            vehicle = self.env.get_vehicle_by_id(v_id)
            capacity = vehicle.capacity_weight + vehicle.capacity_volume
            vehicle_capacities.append((capacity, v_id))
        vehicle_capacities.sort(reverse=True)
        remaining_vehicles = [v_id for _, v_id in vehicle_capacities]
        
        print("\n" + "="*60)
        print("AGGRESSIVE PACKING STRATEGY")
        print("="*60)
        
        iteration = 0
        while remaining_orders and remaining_vehicles:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}: {len(remaining_orders)} orders, {len(remaining_vehicles)} vehicles left")
            
            # Try each vehicle with all remaining orders
            best_assignment = None
            best_order_count = 0
            best_vehicle = None
            
            for vehicle_id in remaining_vehicles:
                # Try to pack as many orders as possible
                packed_orders = self.greedy_pack_orders_to_vehicle(
                    vehicle_id, 
                    list(remaining_orders)
                )
                
                if len(packed_orders) > best_order_count:
                    # Try to create the route
                    test_route = self.create_multi_order_route(vehicle_id, packed_orders)
                    if test_route:
                        best_assignment = test_route
                        best_order_count = len(packed_orders)
                        best_vehicle = vehicle_id
            
            if best_assignment:
                solution['routes'].append(best_assignment)
                self.used_vehicles.add(best_vehicle)
                remaining_vehicles.remove(best_vehicle)
                
                # Update fulfilled orders
                for step in best_assignment['steps']:
                    for delivery in step.get('deliveries', []):
                        order_id = delivery['order_id']
                        fulfilled_orders.add(order_id)
                        remaining_orders.discard(order_id)
                
                print(f"  ✓ {best_vehicle}: {best_order_count} orders packed")
                print(f"  📊 Progress: {len(fulfilled_orders)}/{len(all_orders)} fulfilled")
            else:
                # No valid assignment found - try single orders with any vehicle
                print(f"  ⚠️  No multi-order route possible, trying single orders...")
                
                assigned = False
                for order_id in list(remaining_orders)[:5]:  # Try first 5
                    for vehicle_id in remaining_vehicles:
                        route = self.create_multi_order_route(vehicle_id, [order_id])
                        if route:
                            solution['routes'].append(route)
                            self.used_vehicles.add(vehicle_id)
                            remaining_vehicles.remove(vehicle_id)
                            fulfilled_orders.add(order_id)
                            remaining_orders.discard(order_id)
                            print(f"  ✓ {vehicle_id}: single order {order_id}")
                            assigned = True
                            break
                    if assigned:
                        break
                
                if not assigned:
                    print(f"  ✗ Cannot assign any remaining orders")
                    break
        
        # Final summary
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        print(f"Routes created: {len(solution['routes'])}")
        print(f"Vehicles used: {len(self.used_vehicles)}/{len(all_vehicles)}")
        print(f"Orders fulfilled: {len(fulfilled_orders)}/{len(all_orders)}")
        fulfillment_rate = len(fulfilled_orders)/len(all_orders)*100 if all_orders else 0
        print(f"Fulfillment rate: {fulfillment_rate:.1f}%")
        
        if fulfillment_rate < 100:
            remaining = set(all_orders) - fulfilled_orders
            print(f"\n⚠️  Unfulfilled orders ({len(remaining)}):")
            for oid in list(remaining)[:10]:
                weight, volume, reqs = self.calculate_order_requirements(oid)
                print(f"  - {oid}: {weight:.1f}kg, {volume:.2f}m³, {len(reqs)} SKUs")
            if len(remaining) > 10:
                print(f"  ... and {len(remaining)-10} more")
        else:
            print("\n✅ PERFECT FULFILLMENT ACHIEVED!")
        
        print("=" * 60)
        
        return solution


def solver(env: LogisticsEnvironment) -> Dict:
    """Main solver entry point"""
    optimizer = EnhancedMWVRPSolver(env)
    return optimizer.solve()


# For testing locally (comment out when submitting)
# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     solution = solver(env)