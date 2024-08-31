import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from itertools import permutations
import random
import math
import numpy as np
import time
import heapq
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

Tx  = 33             # dBm 
N0  = 4.002 * 1e-21  # W/Hz 
P_t = 2             # W
c = 3 * 1e8         # m/s 
f = 5 * 1e9         # Hz
cap_sup = 1e7
Gt = 17
Gr = 17
INF = 1e6
MAX_CONSUMPTION = 1000 # MHz
MAX_PENALTY     = 1e6
desired_consumptions  = []
dummy_node = 2000

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"])

def link_capacity(dst, bandwidth, frequency):
    if dst == 0 :
        print("we're having zero")
    denom = N0 * bandwidth * ((4 * np.pi * dst * frequency / c) ** 2)  # corrected
    SNR = (P_t * Gt * Gr) / denom  # simplified
    capacity = (bandwidth * math.log2(1 + SNR)) / 1e6  # converted to Mbps
    return capacity

def spherical_to_cartesian(lat, lon, alt):
    R = 6371.0  # Radius of Earth in km
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    altitude_km = alt / 1000  # Convert meters to kilometers
    x = (R + altitude_km) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (R + altitude_km) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (R + altitude_km) * math.sin(lat_rad)
    return (x, y, z)



def calculate_azimuth_cartesian(x1, y1, x2, y2, x3, y3):
    # Compute vectors AB and AC
    vec_AB = (x2 - x1, y2 - y1)
    vec_AC = (x3 - x1, y3 - y1)
    
    # Compute the dot product
    dot_product = vec_AB[0] * vec_AC[0] + vec_AB[1] * vec_AC[1]
    
    # Compute the magnitudes of the vectors
    mag_AB = math.sqrt(vec_AB[0]**2 + vec_AB[1]**2)
    mag_AC = math.sqrt(vec_AC[0]**2 + vec_AC[1]**2)
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (mag_AB * mag_AC)
    
    # To avoid numerical issues with acos, ensure the value is within the valid range
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # Compute the angle in radians
    theta = math.acos(cos_theta)
    
    # Compute the z-component of the cross product to determine the orientation
    cross_product_z = vec_AB[0] * vec_AC[1] - vec_AB[1] * vec_AC[0]
    
    print("theta",theta)
    # Adjust the angle based on the orientation
    if cross_product_z < 0:
        angle_degrees = -math.degrees(theta)
    else:
        angle_degrees = math.degrees(theta)

    # Normalize angle to handle angles outside the -180 to +180 range
    
    
    return angle_degrees

def calculate_elevation_cartesian(xA, zA, xC, zC, xD, zD):
    # Compute vectors AC and AD in the xz-plane
    vec_AC = (xC - xA, zC - zA)
    vec_AD = (xD - xA, zD - zA)
    
    # Compute the dot product
    dot_product = vec_AC[0] * vec_AD[0] + vec_AC[1] * vec_AD[1]
    
    # Compute the magnitudes of the vectors
    mag_AC = math.sqrt(vec_AC[0]**2 + vec_AC[1]**2)
    mag_AD = math.sqrt(vec_AD[0]**2 + vec_AD[1]**2)
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (mag_AC * mag_AD)
    
    # To avoid numerical issues with acos, ensure the value is within the valid range
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # Compute the angle in radians
    theta = math.acos(cos_theta)
    
    # Convert to degrees
    angle_degrees = math.degrees(theta)
    
    return angle_degrees

def compute_boundaries(n, total_sz, positions, tree):
    upper_bounds_azimuth = [0 for _ in range(total_sz)]
    lower_bounds_azimuth = [0 for _ in range(total_sz)]
    upper_bounds_elevation = [0 for _ in range(total_sz)]
    lower_bounds_elevation = [0 for _ in range(total_sz)]
    azimuth_index = [[-1,-1] for i in range(total_sz)]

    # Convert all positions to Cartesian coordinates
    print(positions[0][0],positions[0][1],positions[0][2])
    positions_cartesian = [spherical_to_cartesian(pos[0], pos[1], pos[2]) for pos in positions]
    print("positions_cartesian",positions_cartesian)
    for f in range(total_sz):
        if len(tree[f])> 0:
            ref_position = positions_cartesian[tree[f][0]]
            azimuth_index[f][1] = tree[f][0]
            azimuth_index[f][0] = tree[f][0]
            print("angle head is", tree[f][0])
            for nxt in tree[f]:
                
                nxt_position = positions_cartesian[nxt]
                azimuth = calculate_azimuth_cartesian(positions_cartesian[f][0], positions_cartesian[f][1],ref_position[0], ref_position[1], nxt_position[0], nxt_position[1])
                elevation = calculate_elevation_cartesian(positions_cartesian[f][0], positions_cartesian[f][1], ref_position[0], ref_position[2],nxt_position[0],nxt_position[2])
                print(tree[f][0],f, nxt ," horiz angle", azimuth )
                upper_bounds_azimuth[f] = max(upper_bounds_azimuth[f], azimuth)
                if upper_bounds_azimuth[f] == azimuth :
                    azimuth_index[f][1] = nxt
                lower_bounds_azimuth[f] = min(lower_bounds_azimuth[f], azimuth)
                if lower_bounds_azimuth[f] == azimuth :
                    azimuth_index[f][0] = nxt
                upper_bounds_elevation[f] = max(upper_bounds_elevation[f], elevation)
                lower_bounds_elevation[f] = min(lower_bounds_elevation[f], elevation)

    return upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation, azimuth_index

def create_interference_list(n, total_sz, positions, tree, node_list, upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation):
    interference_list = [[] for _ in range(total_sz)]
    
    # Convert all positions to Cartesian coordinates
    positions_cartesian = [spherical_to_cartesian(positions[i][0], positions[i][1], positions[i][2]) for i in range(total_sz)]

    for i in range(total_sz):
        if len(tree[i]) > 1:
            ref_position = positions_cartesian[tree[i][0]]
            for j in range(total_sz):
                if j != i and node_list[j]!= -1 and j not in tree[i] :
                    azimuth = calculate_azimuth_cartesian(positions_cartesian[i][0], positions_cartesian[i][1],ref_position[0], ref_position[1], positions_cartesian[j][0], positions_cartesian[j][1])
                    elevation = calculate_elevation_cartesian(positions_cartesian[i][0], positions_cartesian[i][2],ref_position[0], ref_position[2], positions_cartesian[j][0], positions_cartesian[j][2])

                    if (lower_bounds_azimuth[i] <= azimuth <= upper_bounds_azimuth[i] and
                        lower_bounds_elevation[i] <= elevation <= upper_bounds_elevation[i]):
                        interference_list[i].append(j)

    return interference_list

def create_interference_levels(n,total_sz,node_tree, channel_array, interference_list):
    interference_level = [1 for _ in range(total_sz)]
    for i in range(n, total_sz):
        ch = channel_array[i][1]
        for u in interference_list[i]:
            ch1 = channel_array[u][0]
            interference_level[u] += (ch == ch1)
    return interference_level

def capacity_with_interference(n,total_sz,capacity, node_tree, interference_level):
    capacity_cp = [list(lst) for lst in capacity]
    for i in range(total_sz):
        for node in node_tree[i]:
                capacity_cp[i][node] /= (len(node_tree[i]) + interference_level[node])
    return capacity_cp

##########################################################




def nb_hope(n,m,node_tree):
    saut = [0 for _ in range(n + m)]
    visited = [i for i in range(n) ]
    i = 0
    sz = n
    while i < sz:
        f = visited[i]
        i += 1
        for u in node_tree[f]:
            saut[u] = saut[f] + 1
            visited.append(u)
            sz += 1
    return saut

def evaluate(n,m,distances, capacities,desired_consumptions,node_tree, consumed_rates):
    max_capacities = [max(capacities[i]) for i in range(n, n + m)]
    contributions = [0 for _ in range(n + m)]
    saut = nb_hope(n,m,node_tree)
    total_cost = 0
    for prt in range(n + m):
        for u in node_tree[prt]:
            single_cost = 0.6* max((desired_consumptions[u] - consumed_rates[u]) / max(desired_consumptions[u],1) * 100, 1) +0.4 * saut[u] * distances[u][prt]
            if ((consumed_rates[u] == 0 and desired_consumptions[u] !=0) ) :
                 single_cost = MAX_PENALTY
            total_cost += single_cost 
            contributions[u] = single_cost
    return total_cost, contributions

def get_child(node_tree, index1):
    childs = []
    stack = [index1]
    while stack:
        node = stack.pop()
        for child in node_tree[node]:
            childs.append(child)
            stack.append(child)
    return childs

def calculate_total_demand(index,node_tree, desired_consumptions, total_demand):
    total = desired_consumptions[index]  # Start with the demand of the current node
    for child in node_tree[index]:
        total += calculate_total_demand(child, node_tree,  desired_consumptions, total_demand)
    total_demand[index] = total
    return total

def fill_graphs(n,total_sz,positions,distances,capacities,desired_consumptions,node_tree, node_list,channel_array):
    forward_rates = [desired_consumptions[i] if i < n else 0 for i in range(total_sz)]
    input_rates = [0 for i in range(total_sz)]
    consumed_rates = [0 for i in range(total_sz)]
    node_tree1 = [list(lst) for lst in node_tree]
    upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation,azimuth_index = compute_boundaries( n, total_sz, positions, node_tree1 )
    interference_list = create_interference_list(n,total_sz, positions,node_tree1, node_list, upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation)
    interference_levels = create_interference_levels(n,total_sz,node_tree1, channel_array, interference_list)
    capacities_cp = capacity_with_interference(n,total_sz,capacities, node_tree1, interference_levels)
    total_demand = [0 for i in range(total_sz)]
    for i in range(n):
        total = calculate_total_demand( i,node_tree1, desired_consumptions,total_demand)
    for i in range(total_sz):
        node_tree1[i].sort(key=lambda x: distances[i][x])
    visited = [i for i in range(n)]
    i = 0
    sz = n
    while i < sz:
        f = visited[i]
        i += 1
        for u in node_tree1[f]:
            input_rates[u] = min(forward_rates[f], min(total_demand[u], capacities_cp[f][u]))
            consumed_rates[u] = min(input_rates[u], desired_consumptions[u])
            forward_rates[u] = input_rates[u] - consumed_rates[u]
            visited.append(u)
            forward_rates[f] -= input_rates[u]
            sz += 1
    return input_rates, consumed_rates, forward_rates, interference_levels, capacities_cp, azimuth_index

def generate_channels(n,total_sz,positions,available_channels,topology, node_list):
    channel_array = [[0, 0] for _ in range(total_sz)]
    upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation,azimuth_index = compute_boundaries( n, total_sz, positions,topology)
    interference_list = create_interference_list(n,total_sz, positions,topology,node_list, upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation)
    for i in range(total_sz):
        if len(topology[i]) > 0:
                map_ch = {ch: 0 for ch in available_channels }
                map_ch[channel_array[i][0]] = INF 
                for interferent_node in interference_list[i]:
                    if channel_array[interferent_node][0] != 0:
                        ch = channel_array[interferent_node][0]
                        map_ch[ch] += 1
                chosen = min(map_ch, key=map_ch.get)
                
                channel_array[i][1] = chosen
                for child in topology[i]:
                    channel_array[child][0] = chosen
    return channel_array



def build_graph(n,total_sz,distances, possible_connections,total_labels):
    G = nx.Graph() 
    for i in range(total_sz):
        G.add_node(i)
    G.add_node(dummy_node)
    for i in range(n):
        G.add_edge(dummy_node,i,weight = 0)
    for i in range(total_sz):
        for j in possible_connections[i]:
            cost = distances[i][j] 
            if (total_labels[i] ==  'SITEs'  and  total_labels[j] ==  'SITEs'):
                cost += 10000
            if (total_labels[i] ==  'TOTEM_SA'  or  total_labels[j] ==  'TOTEM_SA'): 
                cost +=  100000000
            if (total_labels[i] ==  'SITEs'  or  total_labels[j] ==  'SITEs'):
                cost +=  100
            if (total_labels[i] ==  'RDA'  or  total_labels[j] ==  'RDA'):
                cost +=  3
                
            G.add_edge(i,j,weight = cost)
    return G

def possible_problem(n, total_sz, obligatory_nodes, G):
    visited = [0] * (dummy_node+1)
    def dfs(node):
        stack = [node]
        while stack:
            current = stack.pop()
            print("curretn",current)
            print("neighbors",list(G.neighbors(current)))
            for neighbor in G.neighbors(current):
                print(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)

    visited[dummy_node] = 1
    dfs(dummy_node)
    return visited


def fill_topology(g, total_sz, obligatory_nodes):
    ob_nodes = sorted(obligatory_nodes.copy())
    node_list = [-1 for _ in range(total_sz)]
    node_tree = [[] for _ in range(total_sz)]
    visited = [0 for _ in range(total_sz)]
    
    # Traverse the spanning tree and orient it
    def dfs(node, parent, node_list, node_tree):
        print("node", node)
        visited[node] = 1
        print(g.neighbors(node))
        for neighbor in g.neighbors(node):
            print("neighbor", neighbor)
            if not visited[neighbor]:  # Ensure we only visit unvisited neighbors
                node_tree[node].append(neighbor)
                node_list[neighbor] = node
                dfs(neighbor, node, node_list, node_tree)
    
    # Run DFS for each node in obligatory_nodes if it hasn't been visited
    for root in ob_nodes:
        print("root", root)
        if not visited[root]:
            # Start DFS from the chosen root
            dfs(root, None, node_list, node_tree)
    
    return node_list, node_tree


def generate_steiner_tree(G, obligatory_nodes, total_sz):
    # 1. Identify connected components
    components = list(nx.connected_components(G))
    print(components)
    # Create an empty graph to store the Steiner trees for each component
    induced_subgraph = nx.Graph()
    for i in range(total_sz):
        induced_subgraph.add_node(i)
    induced_subgraph.add_node(dummy_node)
    # 2. Iterate over each connected component
    for component in components:
        subgraph = G.subgraph(component)
        print("subgraph", subgraph)
        # 3. Identify obligatory nodes within the current component
        sub_obligatory_nodes = [node for node in component if node in obligatory_nodes or node == dummy_node]
        
        if len(sub_obligatory_nodes) > 1:    
            # 4. Generate the Steiner tree for the component
            steiner_tree_subgraph = steiner_tree(subgraph, sub_obligatory_nodes, weight='weight')
            
            # 5. Add the Steiner tree edges to the final graph
            induced_subgraph.add_edges_from(steiner_tree_subgraph.edges(data=True)) 
        
    
    return induced_subgraph
        

@app.route('/generate_topology', methods=['POST', 'OPTIONS'])     
def generate_topology():
    try:
        if request.method == 'POST':
            json_data = request.get_json()  # Retrieve JSON data from the request
            print("Received JSON data:", json_data)
            # Ensure all required keys are present in json_data
            required_keys = ['n', 'positions', 'labels','distances', 'obligatory_nodes', 'possible_connections', 'node_capacities','bandwidth','available_channels']
            for key in required_keys:
                if key not in json_data:
                    print(key)
                    return jsonify({'error': f'Missing required key: {key}'}), 400
            # Get the uploaded image from the request
            json_data = request.json
            n = json_data['n'] 
            positions = json_data['positions']
            labels    = json_data['labels']
            distances = json_data['distances']
            possible_connections = json_data['possible_connections']
            obligatory_nodes = json_data['obligatory_nodes']
            desired_consumptions = json_data['node_capacities']
            band_width =  json_data['bandwidth']
            available_channels = json_data['available_channels']
            obligatory_nodes = [i for i in range(n)] + obligatory_nodes        
            total_sz =  len(labels)
            capacities =  [[link_capacity(distances[i][j] * 1000, band_width, f) if i!=j else 0 for j in range(total_sz) ] for i in range(total_sz)]
            g = build_graph(n,total_sz,distances, possible_connections,labels)
            node_tree = [[] for _ in range(total_sz)]
            node_list = [-1 for _ in range(total_sz)]
            channel_array = [[0,0] for _ in range(total_sz)]
            visited = possible_problem(n, total_sz,obligatory_nodes, g)
            sol = all([visited[i] == 1 for i in obligatory_nodes])
            print(sol)
            st = generate_steiner_tree(g,obligatory_nodes,total_sz)
            if dummy_node in st:
                st = st.copy()  # Make sure the Steiner tree is mutable
                st.remove_node(dummy_node)
            print("final_tree",st)
            for u, v, data in st.edges(data=True):
                print(f"Edge ({u}, {v}) with attributes: {data}")
            # Looping over the edges of the Steiner tree
            node_list, node_tree = fill_topology(st,total_sz,obligatory_nodes)
            channel_array = generate_channels(n,total_sz,positions,available_channels,node_tree, node_list)
            input_rates, consumed_rates, forward_rates, interference_levels, capacities_cp, azimuth_index = fill_graphs(n,total_sz,positions,distances,capacities,desired_consumptions,node_tree, node_list,channel_array)
            upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation, azimuth_index = compute_boundaries(n, total_sz, positions, node_tree)
            horizontal_azimuth = [upper - lower for upper, lower in zip(upper_bounds_azimuth, lower_bounds_azimuth)]
            vertical_azimuth = [upper - lower for upper, lower in zip(upper_bounds_elevation, lower_bounds_elevation)]
            response = {'state': 1,
                                'node_list' : node_list ,
                                'node_tree' : node_tree ,
                                'channel_array' :channel_array,
                                'connection_boundaries': azimuth_index,
                                'data_list': input_rates,
                                'consumed_rates'   : consumed_rates,
                                'horizontal_azimuth' : horizontal_azimuth,
                                'vertical_azimuth':vertical_azimuth,
                                'capacities': capacities_cp,
                            }
            print(response)
            response = jsonify(response)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
           
        elif request.method == 'OPTIONS':
            # Handle preflight request
            response_data = {'message': 'Preflight request successful'}
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
            return response
        else:
            return jsonify({'error': 'Method not allowed'}), 405
    except Exception as e:
        print(5)
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
        app.run(port=5000)

