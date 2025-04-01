from geopandas import GeoDataFrame
from shapely.geometry import Point

class ClusterGeoDataframe(GeoDataFrame):
    ''' 
    A subclass of GeoDataFrame that adds clustering functionality.
    This class is designed to handle geospatial data and perform clustering operations.'''
    
    def __init__(self, n_clusters:int, *args, **kwargs):
        """
        Initialize the ClusterGeoDataframe with a number of clusters.
        
        Parameters:
        n_clusters (int): The number of clusters to create.
        *args: Additional positional arguments for GeoDataFrame.
        **kwargs: Additional keyword arguments for GeoDataFrame.
        """
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        # create a column for cluster labels
        self['cluster'] = None
        # create a column for cluster centers
        self['cluster_center'] = None