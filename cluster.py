from geopandas import GeoDataFrame
from sklearn.cluster import KMeans
import numpy as np
from shapely.geometry import Point
import plotly.express as px

def cluster(gdf:GeoDataFrame,n_clusters:int) -> GeoDataFrame:
    """
    Cluster the points in the GeoDataFrame into n_clusters using KMeans clustering.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing the points to be clustered.
        n_clusters (int): The number of clusters to form.

    Returns:
        GeoDataFrame: A new GeoDataFrame with an additional column 'cluster' indicating the cluster each point belongs to.
    """

    # Extract coordinates from the geometry column
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    
    # Create a new column in the GeoDataFrame for the cluster labels
    gdf['cluster'] = kmeans.labels_

    return gdf

def plot_clusters(gdf:GeoDataFrame) -> px.Figure:
    """
    Plot the clusters in the GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing the points and their cluster labels.
    """

    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    
    fig = px.scatter_mapbox(
        gdf,
        lat="lat",
        lon="lon",
        zoom=10,
        mapbox_style="open-street-map",
        color="cluster",
    )

    return fig