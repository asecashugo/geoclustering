from geopandas import GeoDataFrame
from sklearn.cluster import KMeans
import numpy as np
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go

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

def plot_clusters(gdf: GeoDataFrame) -> go.Figure:
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    fig = go.Figure()

    for cluster_id, cluster_df in gdf.groupby("cluster"):
        fig.add_trace(
            go.Scattermap(
                lat=cluster_df["lat"],
                lon=cluster_df["lon"],
                mode="markers",
                marker=dict(size=8),
                name=f"Cluster {cluster_id}",
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=10,
            center=dict(lat=gdf["lat"].mean(), lon=gdf["lon"].mean()),
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig