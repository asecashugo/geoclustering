def test_html_map():

    import geopandas as gpd
    from clustering.cluster import cluster, plot_clusters
    N_CLUSTERS=2
    points_gdf=gpd.GeoDataFrame.from_file("input/points.geojson")
    points_gdf=cluster(points_gdf, N_CLUSTERS)
    fig=plot_clusters(points_gdf)
    import os
    os.makedirs("output", exist_ok=True)
    fig.write_html("output/clustered_points.html")
    assert os.path.exists("output/clustered_points.html")