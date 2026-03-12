#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:14:07 2026

@author: kylel
"""

# %% imports

import os
import pandas as pd
import src.shp_reader as shp_reader

def load_and_subset(filename, cols_map, **kwargs):
    path = os.path.join(os.getcwd(), "data", filename)
    df = pd.read_csv(path, **kwargs)
    df = df[list(cols_map.keys())].rename(columns=cols_map)
    df = df.replace("No Data", pd.NA)
    # .all(axis=1) checks if all values in the row are non-missing
    df = df[df.notna().all(axis=1)] 
    return df

# %% read in shapefile

fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
shp, _ = shp_reader.read_US_shapefile(fp)
shp['County_FIPS'] = (shp['STATEFP'] + shp['COUNTYFP']).astype(int)

# %% load covariate data

unemployment = load_and_subset("DiabetesAtlasData_unemployment_2014.csv", {
    'County_FIPS': 'County_FIPS', 
    'Unemployed-2014-Percentile': 'unemployed_2014'
}, skiprows = 2)

svi = load_and_subset("DiabetesAtlasData_SVI_2014.csv", {
    'County_FIPS': 'County_FIPS', 
    'Overall Socioeconomic Status-2014-Percentile': 'SVI_2014'
}, skiprows = 2)

diabetes = load_and_subset("DiabetesAtlasData_diabetes_2014.csv", {
    'County_FIPS': 'County_FIPS', 
    'Diagnosed Diabetes-2014-Percentage': 'diabetes_2014'
}, skiprows = 2)

obesity = load_and_subset("DiabetesAtlasData_obesity_2014.csv", {
    'County_FIPS': 'County_FIPS', 
    'Obesity-2014-Percentage': 'obesity_2014'
}, skiprows = 2)

inactivity_uninsured = load_and_subset("DiabetesAtlasData_inactivity_uninsured.csv", {
    'County_FIPS': 'County_FIPS', 
    'Physical Inactivity-2014-Percentage': 'inactivity_2014',
    'No Health Insurance-2012-2016-Percentage': 'uninsured_2012_2016'
}, skiprows = 2)

data = pd.merge(unemployment, svi, how = 'inner')
data = pd.merge(data, diabetes, how = 'inner')
data = pd.merge(data, obesity, how = 'inner')
data = pd.merge(data, inactivity_uninsured, how = 'inner')
data['County_FIPS'] = data['County_FIPS'].astype(int)
data = data[data['County_FIPS'].isin(shp['County_FIPS'])]

# %% import smoking data

filename = "IHME_US_COUNTY_TOTAL_AND_DAILY_SMOKING_PREVALENCE_1996_2012.csv"
path = os.path.join(os.getcwd(), "data", filename)
smoking = pd.read_csv(path)
smoking = smoking.query("year == 2012 and sex == 'Both' and county.notnull()")
cols_map = {
    'county': 'county',
    'state': 'state',
    'total_mean': 'total_mean_smoking'}
smoking = smoking[list(cols_map.keys())].rename(columns=cols_map)
smoking['location'] = smoking['county'] + ', ' + smoking['state']
smoking = smoking[smoking.notna().all(axis=1)]

# %% fix smoking

# some manual fixes ...
smoking.loc[
    (smoking['county'] == "Shannon County") & (smoking['state'] == "South Dakota"), 
    'location'
] = "Oglala Lakota County, South Dakota"
smoking.loc[
    (smoking['county'] == "Prince William County/Manassas Park City"), 
    'location'
] = "Prince William County, Virginia"
smoking.loc[
    (smoking['county'] == "Augusta County/Waynesboro City"), 
    'location'
] = "Augusta County, Virginia"
smoking.loc[
    (smoking['county'] == "Bedford County/Bedford City"), 
    'location'
] = "Bedford County, Virginia"
smoking.loc[
    (smoking['county'] == "Fairfax County/Fairfax City"), 
    'location'
] = "Fairfax County, Virginia"
smoking.loc[
    (smoking['county'] == "Southampton County/Franklin City"), 
    'location'
] = "Southampton County, Virginia"
smoking.loc[
    (smoking['county'] == "La Salle Parish"),
    "location"
] = "LaSalle Parish, Louisiana"
smoking.loc[
    (smoking['state'] == "District of Columbia"),
    "location"
] = "District of Columbia, District of Columbia"

# split counties in virginia
newrows2 = pd.DataFrame()
newrows2['county'] = ["Waynesboro City", "Manassas Park City",
                      "Fairfax City", "Franklin City"]
newrows2['state'] = "Virginia"
newrows2['total_mean_smoking'] = [
    smoking.loc[(smoking['county'] == "Augusta County/Waynesboro City"), 'total_mean_smoking'].item(),
    smoking.loc[(smoking['county'] == "Prince William County/Manassas Park City"), 'total_mean_smoking'].item(),
    smoking.loc[(smoking['county'] == "Fairfax County/Fairfax City"), 'total_mean_smoking'].item(),
    smoking.loc[(smoking['county'] == "Southampton County/Franklin City"), 'total_mean_smoking'].item()
]
newrows2['location'] = ["Waynesboro City, Virginia",
                        "Manassas Park City, Virginia",
                        "Fairfax City, Virginia",
                        "Franklin City, Virginia"] 


# split up counties in colorado
indx = (smoking['county'] == "Adams County/Boulder County/Broomfield County/Jefferson County/Weld County")
z = smoking.loc[indx]
newrows = pd.DataFrame()
newrows['county'] = ["Adams County", "Boulder County", "Broomfield County", "Jefferson County", "Weld County"]
newrows['state'] = 'Colorado'
newrows['total_mean_smoking'] = z['total_mean_smoking'].item()
newrows['location'] = newrows['county'] + ', ' + newrows['state']
smoking = pd.concat([smoking.loc[-indx], newrows, newrows2], ignore_index = True)
smoking['location'] = smoking['location'].str.replace("St.", "Saint", regex=False)
smoking['location'] = smoking['location'].str.replace("Ste.", "Sainte", regex=False)

# %% load in response data

cancer = pd.read_excel(os.path.join("data", "IHME_county_cancer_mortality.xlsx"), 
                       sheet_name="Tracheal, bronchus, & lung ", skiprows=1)
cols_map = {
    'Location': 'location',
    'FIPS': 'County_FIPS',
    'Mortality Rate, 2014*': 'mortality2014'}
cancer = cancer[list(cols_map.keys())].rename(columns=cols_map)
cancer = cancer[cancer.notna().all(axis=1)]
cancer['County_FIPS'] = cancer['County_FIPS'].astype(int)
cancer = cancer.query('County_FIPS.notnull() and County_FIPS >= 1000')

pattern = "^(.+)\\s\\("
cancer['mortality2014'] = cancer['mortality2014'].str.extract(pattern).astype(float)

# %% merge

smoking_cancer = pd.merge(cancer, smoking, how = "inner", on = "location")
smoking_cancer['County_FIPS'] = smoking_cancer['County_FIPS'].astype(int)
all_data = pd.merge(data, smoking_cancer, how = "inner", on = "County_FIPS")
#all_data = all_data[all_data['County_FIPS'].isin(shp['County_FIPS'])]

pred_cols = ['total_mean_smoking', 'unemployed_2014', 'SVI_2014', 'inactivity_2014',
'uninsured_2012_2016', 'diabetes_2014', 'obesity_2014']
all_data[pred_cols] = all_data[pred_cols].astype(float)

# %% save 

all_data.to_csv("output/RDA/data_cleaned.csv")

# %% Merge all data

data_shp = pd.merge(shp, all_data, on = "County_FIPS", how = "inner")
data_shp = data_shp.reset_index(drop = True)
print("Evaluating len(data_shp) == len(all_data)...")
print(len(data_shp) == len(all_data))