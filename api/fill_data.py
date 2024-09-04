import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import math

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"])

# Replace with your actual Google Maps API Key
GOOGLE_API_KEY = "AIzaSyBeya4HQeQc6Ps1nROlox3qrQV3fbrJRAc"


def haversine_distance(coord1, coord2):
    # Calculate the distance between two points (lat, lon) using the Haversine formula
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1 = math.radians(coord1['lat']), math.radians(coord1['lng'])
    lat2, lon2 = math.radians(coord2['lat']), math.radians(coord2['lng'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def get_elevation_along_path(loc1, loc2, samples):
    # Request elevation data from the Google Maps Elevation API
    path_str = f"{loc1['lat']},{loc1['lng']}|{loc2['lat']},{loc2['lng']}"
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "path": path_str,
        "samples": samples,
        "key": GOOGLE_API_KEY
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK' and len(data['results']) > 0:
            elevation_data = [result['elevation'] for result in data['results']]
            return elevation_data
    return [0] * samples  # Return default value on error


def get_elevation_path(loc1, loc2,samples):
    elevation_data = get_elevation_along_path(loc1, loc2, samples)
    return jsonify({"elevation": elevation_data})

@app.route('/fill-matrices', methods=['POST','OPTIONS'])
def fill_matrices():
    try:
        if request.method == 'POST':
            data = request.json
            positions_map = data.get('positions_map')
            samples = data.get('samples')

            if not positions_map or not samples:
                return jsonify({"error": "Invalid input"}), 400

            sz = len(positions_map)
            distances = [[0 for _ in range(sz)] for _ in range(sz)]
            path_elevations = [[[] for _ in range(sz)] for _ in range(sz)]

            for i in range(sz):
                print(i)
                for j in range(i + 1, sz):
                    dst = haversine_distance(positions_map[i], positions_map[j])
                    distances[i][j] = dst
                    distances[j][i] = dst
                    elev_path = get_elevation_along_path(positions_map[i], positions_map[j], samples) if dst < 15 else [0] * samples
                    path_elevations[i][j] = elev_path
                    path_elevations[j][i] = elev_path[::-1]  # Reverse the elevation path for the opposite direction
            print("distances", distances,"path_elevations", path_elevations)
            return jsonify({
                "distances": distances,
                "path_elevations": path_elevations
            })
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
    app.run(debug=True,port = 3000)