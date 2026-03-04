#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 15:10:20 2026

@author: kylel
"""


import geopandas as gpd
from libpysal.weights import Rook, fuzzy_contiguity
import networkx as nx
from shapely.geometry import Polygon


def read_US_shapefile(fp):
    # read in data
    county_shp = gpd.read_file(fp)
    us_shp = county_shp.query("STATEFP not in ['02', '15', '60', '66', '69', '72', '78']")

    # Albers Equal Area Conic projection, units: meters
    us_shp = us_shp.to_crs(5070)
    # reset index
    us_shp = us_shp.reset_index(drop=True)

    # obtain full Rook contiguity adjacency matrix
    W_full = Rook.from_dataframe(us_shp, use_index = False, silence_warnings = True)

    # find largest connected cluster (mainland US)
    G_full = W_full.to_networkx()
    largest_cc_nodes = max(nx.connected_components(G_full), key=len)
    us_mainland = us_shp.iloc[list(largest_cc_nodes)]

    # identify all indices in the original filtered GeoDataFrame
    all_indices = set(range(len(us_shp)))

    # identify indices in the largest cluster
    largest_cluster_indices = set(largest_cc_nodes)

    # find the difference (the "Islands")
    excluded_indices = list(all_indices - largest_cluster_indices)

    # create a DataFrame of the excluded counties
    excluded_counties = us_shp.iloc[excluded_indices]

    # print the names and states
    print(f"Total counties excluded: {len(excluded_counties)}")
    print(excluded_counties[['NAME', 'STATEFP']].sort_values(by='NAME'))

    # reset weights and get adjancency matrix of mainland US
    us_mainland = us_mainland.reset_index(drop = True)
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

if __name__ == "__main__":
    # set filepath to US county shapefile
    fp = "../data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
    us_mainland, W = read_US_shapefile(fp)

