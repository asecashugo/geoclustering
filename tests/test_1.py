import geopandas as gpd
import cluster
N_CLUSTERS=2
points_gdf=gpd.GeoDataFrame.from_file("input/points.geojson")
points_gdf=cluster.cluster(points_gdf, N_CLUSTERS)
fig=cluster.plot_clusters(points_gdf)
fig.write_html("output/clustered_points.html")