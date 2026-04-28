#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kylel
"""


import geopandas as gpd
from libpysal.weights import Rook
import networkx as nx

def exclude_islands(shp):
    # reset index
    shp = shp.reset_index(drop=True)

    # obtain full Rook contiguity adjacency matrix
    W_full = Rook.from_dataframe(shp, use_index = False, silence_warnings = True)

    # find largest connected cluster (mainland US)
    G_full = W_full.to_networkx()
    largest_cc_nodes = max(nx.connected_components(G_full), key=len)
    shp_subset = shp.iloc[list(largest_cc_nodes)]

    # identify all indices in the original filtered GeoDataFrame
    all_indices = set(range(len(shp)))

    # identify indices in the largest cluster
    largest_cluster_indices = set(largest_cc_nodes)

    # find the difference (the "Islands")
    excluded_indices = list(all_indices - largest_cluster_indices)

    # create a DataFrame of the excluded counties
    excluded_counties = shp.iloc[excluded_indices]

    # print the names and states
    print(f"Total counties excluded: {len(excluded_counties)}")
    print(excluded_counties[['NAME', 'STATEFP']].sort_values(by='NAME'))
    shp_subset = shp_subset.reset_index(drop = True)
    return shp_subset

def read_US_shapefile(fp):
    # read in data
    county_shp = gpd.read_file(fp)
    us_shp = county_shp.query("STATEFP not in ['02', '15', '60', '66', '69', '72', '78']")

    # Albers Equal Area Conic projection, units: meters
    us_shp = us_shp.to_crs(5070)

    # reset weights and get adjancency matrix of mainland US
    us_mainland = exclude_islands(us_shp)
    W = Rook.from_dataframe(us_mainland, use_index = False)
    
    return us_mainland, W

def read_CA_shapefile(fp):
    # read in data
    county_shp = gpd.read_file(fp)
    ca_shp = county_shp.query("STATEFP in ['06']")

    # Albers Equal Area Conic projection, units: meters
    ca_shp = ca_shp.to_crs(5070)
    # reset index
    ca_shp = ca_shp.reset_index(drop=True)

    W = Rook.from_dataframe(ca_shp, use_index = False)
    
    return ca_shp, W

def read_CAORWA_shapefile(fp):
    # read in data
    county_shp = gpd.read_file(fp)
    shp = county_shp.query("STATEFP in ['06', '41', '53']")

    # Albers Equal Area Conic projection, units: meters
    shp = shp.to_crs(5070)
    # reset index
    shp = shp.reset_index(drop=True)
    # remove islands
    shp_subset = exclude_islands(shp)
    W = Rook.from_dataframe(shp_subset, use_index = False)
    
    return shp_subset, W
    


if __name__ == "__main__":
    # set filepath to US county shapefile
    fp = "../data/cb_2014_us_county_500k/cb_2014_us_county_500k.shp"
    us_mainland, W = read_US_shapefile(fp)

