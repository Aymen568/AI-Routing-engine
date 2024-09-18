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


desired_consumptions  = []
dummy_node = 2000

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"])


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

##########################################################


def get_child(node_tree, index1):
    childs = []
    stack = [index1]
    while stack:
        node = stack.pop()
        for child in node_tree[node]:
            childs.append(child)
            stack.append(child)
    return childs

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
                cost += 100
            else:
                if (total_labels[i] ==  'TOTEM-SA'  or  total_labels[j] ==  'TOTEM-SA'): 
                    cost +=  10000
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
            required_keys = ['n', 'positions', 'labels','distances', 'obligatory_nodes', 'possible_connections']
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
            obligatory_nodes = [i for i in range(n)] + obligatory_nodes        
            total_sz =  len(labels)
            g = build_graph(n,total_sz,distances, possible_connections,labels)
            node_tree = [[] for _ in range(total_sz)]
            node_list = [-1 for _ in range(total_sz)]
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
            upper_bounds_azimuth, lower_bounds_azimuth, upper_bounds_elevation, lower_bounds_elevation, azimuth_index = compute_boundaries(n, total_sz, positions, node_tree)
            horizontal_azimuth = [upper - lower for upper, lower in zip(upper_bounds_azimuth, lower_bounds_azimuth)]
            vertical_azimuth = [upper - lower for upper, lower in zip(upper_bounds_elevation, lower_bounds_elevation)]
            response = {'state': 1,
                                'node_list' : node_list ,
                                'node_tree' : node_tree ,
                                'connection_boundaries': azimuth_index,
                                'horizontal_azimuth' : horizontal_azimuth,
                                'vertical_azimuth':vertical_azimuth,
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

