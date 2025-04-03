def test_full_html_map():

    import geopandas as gpd
    from clustering.cluster import cluster
    from clustering.connect import get_centroids_and_trees
    N_CLUSTERS=2
    points_gdf=gpd.GeoDataFrame.from_file("input/points.geojson")
    points_gdf=cluster(points_gdf, N_CLUSTERS)
    clusters_gdf, paths_gdf,fig = get_centroids_and_trees(points_gdf)
    import os
    os.makedirs("output", exist_ok=True)
    fig.write_html("output/fig.html")
    assert os.path.exists("output/fig.html")