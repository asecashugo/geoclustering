from geopandas import GeoDataFrame
import numpy as np
import itertools
import copy
from shapely.geometry import Point, Polygon
import plotly.graph_objects as go
from scipy.spatial import Voronoi

def get_cost_matrix(gdf: GeoDataFrame) -> np.ndarray:
    """
    Calculate the cost matrix for the points in the GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing the points.

    Returns:
        np.ndarray: A 2D numpy array representing the cost matrix.
    """
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    cost_matrix = np.zeros((len(coords), len(coords)))

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            # calculate cost only for points in the same cluster
            if gdf.loc[i, 'cluster'] == gdf.loc[j, 'cluster']:
                cost_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
                cost_matrix[j][i] = cost_matrix[i][j]
            else:
                cost_matrix[i][j] = np.inf
                cost_matrix[j][i] = np.inf

    return cost_matrix

class Graph_Dijkstra(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)
    
    def construct_graph(self, nodes, init_graph):
        graph = {}
        for node in nodes:
            graph[node] = {}
        
        graph.update(init_graph)
        
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
                    
        return graph
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]

def dijkstra_algorithm(graph:Graph_Dijkstra, start_node):
    unvisited_nodes = list(graph.get_nodes())
 
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {}
 
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
 
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = np.inf
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = 0
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path

def get_paths(graph, start_node, bonus_factor=0.95):
    
    # print(f'Getting paths disctount factor {bonus_factor}...')
    
    # Initialize the list to hold paths
    paths = []

    # Run Dijkstra's algorithm once to get initial rankings based on hops
    previous_nodes, _ = dijkstra_algorithm(graph, start_node)
    
    # Calculate number of hops for each node
    hop_counts = {}
    for node in graph.get_nodes():
        hops = 0
        temp_node = node
        while temp_node != start_node:
            temp_node = previous_nodes[temp_node]
            hops += 1
        hop_counts[node] = hops

    # Sort nodes by number of hops (furthest first)
    sorted_nodes = sorted(hop_counts, key=hop_counts.get, reverse=True)
    
    for node in sorted_nodes:
        path = []
        temp_node = node
        while temp_node != start_node:
            path.append(temp_node)
            temp_node = previous_nodes[temp_node]
        path.append(start_node)
        paths.append(path[::-1])  # Reverse to have paths from start_node -> node

        # Update graph costs along this path by multiplying by the discount factor
        for i in range(len(path) - 1):
            old_cost = graph.value(path[i], path[i + 1])
            new_cost = old_cost * bonus_factor
            graph.graph[path[i]][path[i + 1]] = new_cost
            graph.graph[path[i + 1]][path[i]] = new_cost  # Assuming undirected graph

        # Recompute Dijkstra's for the next iteration
        previous_nodes, _ = dijkstra_algorithm(graph, start_node)

    return paths

def delaunay_triangulation(gdf: GeoDataFrame) -> GeoDataFrame:
    """
    Perform Delaunay triangulation on the points in the
    GeoDataFrame."
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import LineString

    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    delaunay = Delaunay(coords)

    # create 'from' and 'to' columns for the triangulation lines
    # and create LineString objects for each edge in the triangulation

    lines = []
    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            line = LineString([coords[simplex[i]], coords[simplex[(i + 1) % len(simplex)]]])
            lines.append(line)

    # Create a new GeoDataFrame with the triangulation lines
    triangulation_gdf = GeoDataFrame(columns=['cluster','from','to','geometry','cost'], geometry=lines, crs=gdf.crs)
    for index,row in triangulation_gdf.iterrows():
        from_point = Point(row.geometry.coords[0])
        to_point= Point(row.geometry.coords[1])
        from_point_index=int(gdf[gdf.geometry==from_point].index[0])
        to_point_index=int(gdf[gdf.geometry==to_point].index[0])
        cluster_from=gdf[gdf.geometry==from_point]['cluster'].values[0]
        cluster_to=gdf[gdf.geometry==to_point]['cluster'].values[0]
        if cluster_from!=cluster_to:
            continue
        else:
            triangulation_gdf.at[index,'from']=from_point_index
            triangulation_gdf.at[index,'to']=to_point_index
            triangulation_gdf.at[index,'cluster']=cluster_from
            triangulation_gdf.at[index,'cost']=np.linalg.norm(coords[from_point_index]-coords[to_point_index])

    return triangulation_gdf

def get_path_index(from_index:int,to_index:int,paths_gdf:GeoDataFrame) -> int:
    """
    Get the index of the path in the paths_gdf DataFrame based on the from and to indexes.
    """
    # Check if the path exists in the DataFrame
    path_df=paths_gdf[(paths_gdf['from']==from_index) & (paths_gdf['to']==to_index)]
    if len(path_df.index)==1:
        return path_df.index[0]
    else:
        # check if the reverse path exists
        path_df=paths_gdf[(paths_gdf['from']==to_index) & (paths_gdf['to']==from_index)]
        if len(path_df.index)==1:
            return path_df.index[0]
        else:
            raise ValueError(f'Path from {from_index} to {to_index} not found in paths_gdf') 

def calculate_voronoi_polygons(gdf: GeoDataFrame, convex_hull: Polygon) -> GeoDataFrame:
    """
    Calculate Voronoi polygons for each point within the convex hull.
    """
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    vor = Voronoi(coords)
    polygons = []

    for region_index in vor.point_region:
        vertices = vor.regions[region_index]
        if -1 not in vertices:  # Ignore infinite regions
            polygon = Polygon([vor.vertices[i] for i in vertices])
            if polygon.intersects(convex_hull):
                polygons.append(polygon.intersection(convex_hull))
            else:
                polygons.append(None)
        else:
            polygons.append(None)

    gdf['voronoi'] = polygons
    return gdf

def get_centroids_and_trees(gdf: GeoDataFrame, bonus_factor:float=0.95):
        
    n_nodes=gdf['cluster'].nunique()
    
    # CALCULATE COST MATRIX
    print('Calculating cost matrix...')
    cost_matrix=get_cost_matrix(gdf)
    
    # CREATE PATHS DATAFRAME
    print('Triangulating...')
    paths_gdf=delaunay_triangulation(gdf)
    paths_gdf['cluster_by_cost_selected']=-1
    paths_gdf['cluster_by_cost_selected_count']=0

    # FINDING CENTROIDS
    centroids=[]
    n_clusters=max(gdf['cluster'])+1
    print(f'Finding {n_clusters} centroids...')
    clusters_gdf=GeoDataFrame(columns=['n_points','centroid','centroid_index','voronoi','paths'])
    for cluster in range(n_clusters): # zero-based cluster number
        # get points in cluster
        total_cost_by_centroid=[]
        cluster_points_gdf=gdf[gdf['cluster']==cluster]
        n_cluster_points=len(cluster_points_gdf.index)
        print(f' Cluster {cluster} has {n_cluster_points} points')

        clusters_gdf.at[cluster,'n_points']=n_cluster_points
        # reset index of points_gdf
        cluster_points_gdf.reset_index(inplace=True)
        # find centroid
        # loop through wtgs in cluster
        center_point_id=0
        for index,wtg in cluster_points_gdf.iterrows():
            # calculate total cost of all selected paths summed up
            total_cost=0
            # print(f'  Trying {wtg["name"]} as centroid for cluster {node}...',end=' ')
            # define graph
            points=range(len(cluster_points_gdf.index))
            # get list of segments: [index of from wtg, index of to wtg]
            segments=[]
            possible_paths_node_gdf=paths_gdf[paths_gdf['cluster']==cluster]
            # get subset of cost_matrix using possible_paths_node indexes
            mask=np.zeros_like(cost_matrix,dtype=bool)
            mask[np.ix_(cluster_points_gdf['index'],cluster_points_gdf['index'])]=True
            # Apply the mask to the cost matrix
            cost_matrix_subset=cost_matrix[mask].reshape(len(cluster_points_gdf['index']),len(cluster_points_gdf['index']))

            for index,path in possible_paths_node_gdf.iterrows():
                # rel_index_from=cluster_points_gdf[cluster_points_gdf['name']==path['from']].inde x.values[0]
                # rel_index_to=cluster_points_gdf[cluster_points_gdf['name']==path['to']].index.values[0]
                rel_index_from=[cluster_points_gdf[cluster_points_gdf['index']==path['from']]][0].index.values[0]
                rel_index_to=[cluster_points_gdf[cluster_points_gdf['index']==path['to']]][0].index.values[0]
                segments.append([rel_index_from,rel_index_to])
            
            # sort points within segments
            segments=[sorted(segment) for segment in segments]

            # remove duplicates
            segments.sort()
            segments=list(k for k,_ in itertools.groupby(segments))
            # print(segments)

            # get mini cost dictionary from self.cost_matrix
            mini_cost_dict={}
            for point in points:
                mini_cost_dict[point]={}
            for a,b in segments:
                mini_cost_dict[a][b]=cost_matrix_subset[a][b]
                    
                
            
            graph = Graph_Dijkstra(points, mini_cost_dict)
            # compute shortest paths
            paths=get_paths(graph,center_point_id,bonus_factor)
            # create line collection from paths
            for path in paths:
                if len(path)>1:
                    for j in range(len(path)-1) :
                        point_rel_from=path[j]
                        point_rel_to=path[j+1]
                        point_abs_from=cluster_points_gdf.iloc[point_rel_from]['index']
                        point_abs_to=cluster_points_gdf.iloc[point_rel_to]['index']
                        # find path in paths_df
                        # from_to_path_df=paths_gdf[(paths_gdf['from']==point_abs_from) & (paths_gdf['to']==point_abs_to)]
                        # if len(from_to_path_df.index)==1:
                        #     path_index=from_to_path_df.index[0]
                        # else:
                        #     to_from_path_df=paths_gdf[(paths_gdf['from']==point_abs_from) & (paths_gdf['to']==point_abs_to)]
                        #     if len(to_from_path_df.index)==1:
                        #         path_index=to_from_path_df.index[0]
                        path_index=get_path_index(point_abs_from,point_abs_to,paths_gdf)
                        total_cost+=paths_gdf.loc[path_index,'cost']
            # print(round(total_cost))
            # save total_cost
            total_cost_by_centroid.append(total_cost)
            center_point_id+=1

        # find minimum total cost
        min_cost=min(total_cost_by_centroid)
        min_cost_index=total_cost_by_centroid.index(min_cost)

        # recalculate paths with minimum cost
        paths=get_paths(graph,min_cost_index,bonus_factor)
        
        # CALCULATE LINE WIDTHS
        # create line collection from paths
        lines=[]
        for path in paths:
            if len(path)>1:
                for i in range(len(path)-1) :
                    point_A=cluster_points_gdf.iloc[path[i]]
                    point_B=cluster_points_gdf.iloc[path[i+1]]
                    lines.append([(point_A.geometry.y, point_A.geometry.x),(point_B.geometry.y, point_B.geometry.x)])
                    # store node info in cluster_by_cost_selected
                    path_index=get_path_index(point_A['index'],point_B['index'],paths_gdf)
                    paths_gdf.loc[path_index,'cluster_by_cost_selected']=cluster
                    paths_gdf.loc[path_index,'cluster_by_cost_selected_count']+=1

        lines.sort()

        # remove duplicates
        lines=list(k for k,_ in itertools.groupby(lines))

        # create line collection from paths
        for path in paths:
            if len(path)>1:
                for j in range(len(path)-1) :
                    point_rel_from=path[j]
                    point_rel_to=path[j+1]
                    point_abs_from=cluster_points_gdf.iloc[point_rel_from]['index']
                    point_abs_to=cluster_points_gdf.iloc[point_rel_to]['index']
                    path_index=get_path_index(point_abs_from,point_abs_to,paths_gdf)
                    total_cost+=paths_gdf.loc[path_index,'cost']
        # create copy of paths using absolute indexes

        paths_abs=copy.deepcopy(paths)
        for path in paths_abs:
            for i in range(len(path)):
                path[i]=cluster_points_gdf.iloc[path[i]]['index']
        clusters_gdf.at[cluster,'paths']=paths_abs


        # set centroid
        centroid_index=cluster_points_gdf.iloc[min_cost_index]['index']

        # save centroid
        clusters_gdf.at[cluster,'centroid']=cluster_points_gdf.iloc[min_cost_index]['geometry']
        clusters_gdf.at[cluster,'centroid_index']=centroid_index

        print(f'  Centroid for cluster {cluster} is {centroid_index} with cost {min_cost:.2f}')
        centroids.append(centroid_index)

    # check length of trees
    if len(paths_gdf[paths_gdf.cluster_by_cost_selected!=-1])==len(gdf)-n_nodes:
        print(f'Count OK: {len(paths_gdf[paths_gdf.cluster_by_cost_selected!=-1])} paths for {len(gdf)-1} points in {n_nodes} nodes')
    else:
        print(f'⚠️ Count NOT OK: {len(paths_gdf[paths_gdf.cluster_by_cost_selected!=-1])} paths for {len(gdf)-1} points in {n_nodes} nodes')
    
    # Create a Mapbox figure
    fig = go.Figure()

    # Add line elements for each cluster with a different color
    # for cluster_id, cluster_df in clusters_gdf.groupby("cluster"):
    for cluster_id  in clusters_gdf.index:
        cluster_paths = paths_gdf[paths_gdf["cluster"] == cluster_id]
        showlegend = True
        for _, path in cluster_paths.iterrows():
            if path["cluster_by_cost_selected_count"] > 0:
                line_coords = list(path.geometry.coords)
                fig.add_trace(
                    go.Scattermapbox(
                        lon=[coord[0] for coord in line_coords],
                        lat=[coord[1] for coord in line_coords],
                        mode="lines",
                        line=dict(width=int(path["cluster_by_cost_selected_count"]**0.3), color=f"hsl({cluster_id * 360 / len(clusters_gdf)}, 70%, 50%)"),
                        name=f"Cluster {cluster_id}",
                        legendgroup=f"Cluster {cluster_id}",
                        showlegend=showlegend
                    )
                )
                showlegend = False

    # Calculate the convex hull of all points
    print("Calculating convex hull...")
    convex_hull = gdf.unary_union.convex_hull

    # Ensure the convex hull is a Polygon
    if convex_hull.geom_type == "Point" or convex_hull.geom_type == "MultiPoint":
        # Create a small buffer around the point(s) to form a polygon
        convex_hull = convex_hull.buffer(0.001)

    # Add the convex hull to the figure
    fig.add_trace(
        go.Scattermapbox(
            lon=[coord[0] for coord in convex_hull.exterior.coords],
            lat=[coord[1] for coord in convex_hull.exterior.coords],
            mode="lines",
            fill="toself",
            fillcolor="rgba(128, 128, 128, 0.3)",  # Transparent gray
            line=dict(width=0),
            name="Area",
            showlegend=True,
        )
    )

    # Calculate Voronoi polygons
    print("Calculating Voronoi polygons...")
    gdf = calculate_voronoi_polygons(gdf, convex_hull)

    # Combine convex hulls for each cluster
    for cluster_id in clusters_gdf.index:
        cluster_points = gdf[gdf["cluster"] == cluster_id]
        cluster_convex_hull = cluster_points.unary_union.convex_hull
        clusters_gdf.at[cluster_id, "voronoi"] = cluster_convex_hull  # Replace 'voronoi' with convex hull

        # Add the cluster convex hull to the figure
        if cluster_convex_hull and cluster_convex_hull.geom_type in ["Polygon", "MultiPolygon"]:
            if cluster_convex_hull.geom_type == "Polygon":
                polygons = [cluster_convex_hull]
            else:  # MultiPolygon
                polygons = list(cluster_convex_hull)

            for polygon in polygons:
                fig.add_trace(
                    go.Scattermapbox(
                        lon=[coord[0] for coord in polygon.exterior.coords],
                        lat=[coord[1] for coord in polygon.exterior.coords],
                        mode="lines",
                        fill="toself",
                        # fillcolor=f"hsl({cluster_id * 360 / len(clusters_gdf)}, 70%, 50%)",
                        # use color above but with transparency
                        fillcolor=f"hsla({cluster_id * 360 / len(clusters_gdf)}, 70%, 50%, 0.3)",
                        line=dict(width=0),
                        name=f"Cluster {cluster_id} Convex Hull",
                        showlegend=False,
                        legendgroup="Hulls",
                    )
                )

    # Add all points in grey under the "Points" legend group
    fig.add_trace(
        go.Scattermapbox(
            lon=gdf.geometry.x,
            lat=gdf.geometry.y,
            mode="markers",
            marker=dict(size=8, color="grey"),
            name="Points",
            legendgroup="Points",
            showlegend=True,
        )
    )

    # Add points colored by cluster under the "Clusters" legend group
    for cluster_id in clusters_gdf.index:
        cluster_points = gdf[gdf["cluster"] == cluster_id]
        fig.add_trace(
            go.Scattermapbox(
                lon=cluster_points.geometry.x,
                lat=cluster_points.geometry.y,
                mode="markers",
                marker=dict(size=8, color=f"hsl({cluster_id * 360 / len(clusters_gdf)}, 70%, 50%)"),
                name=f"Cluster {cluster_id}",
                legendgroup="Clusters",
                showlegend=True,
            )
        )

    # Update the layout for the Mapbox figure
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=60,
            center=dict(
                lat=gdf.geometry.y.mean(),
                lon=gdf.geometry.x.mean(),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Set the mapbox style to "carto-positron"
    fig.update_layout(mapbox_style="carto-positron")

    # Return clusters_gdf, paths_gdf, and the figure
    return clusters_gdf, paths_gdf, fig


