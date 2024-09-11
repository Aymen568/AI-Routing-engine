# <p align="center" style="font-size: 60px;"><strong>Comprehensive Topology Planner</strong> </p> 

<p align="center">
  <a href="#overview">Overview</a> |  
  <a href="#technologies-and-architecture">Technologies & Architecture</a> | 
  <a href="#workflow">Workflow</a> | 
  <a href="#functionalities">Functionalities</a>
</p>

## 🔗 Overview

This project focuses on optimizing network topologies by managing a variety of complex node types (such as RDA, Internet Sources, and Totem Points) while ensuring that specific categories of nodes maintain reachability to the internet. The comprehensive topology planner enables efficient planning, visualization, and optimization for network infrastructures with different node characteristics and requirements.

## 🔗 Technologies & Architecture

The architecture diagram below illustrates the data flow and integration of technologies within this project:

<p align="center">
  <img src="/media/github/tools.png" alt="Architecture" width="600" />
</p>

- **Frontend (Client-Side):** Developed using HTML, CSS, and JavaScript, the front end provides an interactive user interface for data input, visualization, and decision-making. The Google Maps JavaScript API is a key component, allowing for the display of interactive maps, adding markers for network nodes, and visualizing potential connections.

- **Backend (Server-Side):** Built using the Flask framework in Python, the backend handles data processing, optimization logic, and communication with external services to perform complex computations and network optimizations.

- **Data Services Integration:** The Google Maps JavaScript API also serves as a data retrieval tool, enabling the front end to fetch geospatial data from Google’s servers. For example, it can call the Elevation Service to retrieve elevation profiles based on geographic coordinates. These requests are initiated from the sever side and processed by Google’s backend services, with the results handled within the front-end JavaScript environment.

## 🔗 Workflow

The application is designed for a specific use case. To achieve the desired results, the KML input file must follow a predefined structure:

<p align="center">
  <img src="/media/github/directories.png" alt="Directory Structure" width="300" />
</p>

Each network node will have the following characteristics:

<p align="center">
  <img src="/media/github/characteristics.png" alt="Node Characteristics" width="400" />
</p>

The overall workflow of the application is illustrated below:

<p align="center">
  <img src="/media/github/workflow.png" alt="Application Workflow" width="400" />
</p>

### Backend Workflow

- **`fill_matrices.py`:** This API retrieves elevation profiles for each possible link by interacting with the Google Maps API.

- **`graph.py`:** This primary API is responsible for designing the network topology and minimizing the number of Totem points used.

Detailed backend workflow:

<p align="center">
  <img src="/media/github/workflow_backend.png" alt="Backend Workflow" width="300" />
</p>

1. **Process Input Data:** Read and parse the KML file to extract node information.
2. **Compute Possible Links:** Calculate links between nodes that do not exceed the specified threshold.
3. **Customize Link Costs:** Adjust link costs to prioritize desired connections (e.g., better connections between Sites and Internet Sources or RDA).
4. **Add Virtual Node:** Introduce a virtual node to connect all source nodes with zero cost.
5. **Identify Connected Components:** Detect all connected components in the network.
6. **Apply Steiner Tree Algorithm:** Optimize connections within each connected component using the Steiner Tree algorithm.
7. **Generate Output List:** Convert the optimized network into a list format for further analysis or visualization.


### Prerequisites

Ensure all required libraries are installed by running:
```bash
pip install -r requirements.txt

#### 1-  Prerequisites:
preuisit librairies are put under the requiremernts.txt file.
Just run pip install -r requirements.txt

#### 2-START THE BACKEND:
To start the backend and handle requests, navigate to the /API directory and run:
```bash 
python ./fill_data.py
python ./graph.py

##  End Results

The Comprehensive Topology Planner provides an efficient solution for network optimization, enabling users to visualize and plan complex network topologies with diverse node types. By employing advanced algorithms and integrating geospatial data, the tool ensures optimal connectivity and performance. The final output includes an optimized network topology that meets specified criteria, facilitating enhanced decision-making and improved network management.

For further details or to contribute to the project, please refer to the documentation or reach out to the project maintainers.
