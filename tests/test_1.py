import geopandas as gpd
from clustering.cluster import cluster, plot_clusters
N_CLUSTERS=2
points_gdf=gpd.GeoDataFrame.from_file("input/points.geojson")
points_gdf=cluster(points_gdf, N_CLUSTERS)
fig=plot_clusters(points_gdf)
fig.write_html("output/clustered_points.html")