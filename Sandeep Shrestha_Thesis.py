#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import glob
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[2]:


#Loading GIS Shapefile of India
shapefile_path = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
india_map = gpd.read_file(shapefile_path)


# In[3]:


india_map = gpd.read_file(shapefile_path)
india_map.plot()


# In[4]:


plt.show()
india_map.head() 


# In[5]:


# Customizing the plot.
india_map.plot(edgecolor='black', color='lightblue', figsize=(10, 10))

plt.title('Map of Indian States')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[6]:


print(india_map['st_nm'])


# In[7]:


# Filtering to show only a specific state.
Telangana_map = india_map[india_map['st_nm'] == 'Telangana']

# Ploting the filtered map.
Telangana_map.plot(edgecolor='black', color='green')
plt.title('Map of Telangana')
plt.show()


# In[8]:


# Loading AQI site CSV file.
site_data_path = r"C:\Users\Legion\.jupyter\thesis\site.csv" 
site_df = pd.read_csv(site_data_path)

# Display first few rows to verify correct loading
print(site_df.head())


# In[9]:


# Loading AQI site CSV file.
aqi_data_path = r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv" 
aqi_df = pd.read_csv(aqi_data_path)

# Display first few rows to verify correct loading
print(aqi_df.head())


# In[10]:


# Converting site data into a GeoDataFrame.
site_gdf = gpd.GeoDataFrame(site_df, geometry=gpd.points_from_xy(site_df.Longitude, site_df.Latitude), crs="EPSG:4326")

# Loading India shapefile and filter Telangana.
shapefile_path = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
india_map = gpd.read_file(shapefile_path)
telangana_map = india_map[india_map["st_nm"] == "Telangana"]

# Ploting Telangana Map.
fig, ax = plt.subplots(figsize=(10, 8))
telangana_map.plot(ax=ax, color="lightgrey", edgecolor="black")


# Ploting AQI Site Locations.
site_gdf.plot(ax=ax, markersize=20, color="blue", alpha=0.8, edgecolor="black", marker="o", label="AQI Sites")

# Customizing the map.
plt.title("Air Quality Index Monitoring Sites in Telangana")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()


# In[11]:


# Defineing primary file paths (loading from all subdirectories).
primary_folder = r"C:\Users\Legion\.jupyter\thesis\primary"
primary_train_files = glob.glob(primary_folder + r"\d*\train.csv")
primary_test_files = glob.glob(primary_folder + r"\d*\test.csv")
primary_value_files = glob.glob(primary_folder + r"\d*\value.csv")

# Defineing synthetic file paths (with correct filenames).
synthetic_folder = r"C:\Users\Legion\.jupyter\thesis\synthetic"
synthetic_train_path = synthetic_folder + r"\synthetic_train_data.csv"
synthetic_test_path = synthetic_folder + r"\synthetic_test_data.csv"
synthetic_value_path = synthetic_folder + r"\synthetic_value_data.csv"

# Loading primary datasets (from multiple folders).
primary_train_df = pd.concat([pd.read_csv(f) for f in primary_train_files], ignore_index=True)
primary_test_df = pd.concat([pd.read_csv(f) for f in primary_test_files], ignore_index=True)
primary_value_df = pd.concat([pd.read_csv(f) for f in primary_value_files], ignore_index=True)

# Load synthetic datasets with correct filenames
synthetic_train_df = pd.read_csv(synthetic_train_path)
synthetic_test_df = pd.read_csv(synthetic_test_path)
synthetic_value_df = pd.read_csv(synthetic_value_path)

# Ensureing column consistency.
assert list(primary_train_df.columns) == list(synthetic_train_df.columns), "Train Data Columns Do Not Match!"
assert list(primary_test_df.columns) == list(synthetic_test_df.columns), "Test Data Columns Do Not Match!"
assert list(primary_value_df.columns) == list(synthetic_value_df.columns), "Value Data Columns Do Not Match!"

# Combining primary and synthetic datasets.
combined_train_df = pd.concat([primary_train_df, synthetic_train_df], ignore_index=True)
combined_test_df = pd.concat([primary_test_df, synthetic_test_df], ignore_index=True)
combined_value_df = pd.concat([primary_value_df, synthetic_value_df], ignore_index=True)

# Ensureing timestamps are sorted properly.
combined_train_df = combined_train_df.sort_values(by="timestamp").reset_index(drop=True)
combined_test_df = combined_test_df.sort_values(by="timestamp").reset_index(drop=True)
combined_value_df = combined_value_df.sort_values(by="timestamp").reset_index(drop=True)

# Saving and display first few rows for verification.
combined_train_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\combined_train.csv", index=False)
combined_test_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\combined_test.csv", index=False)
combined_value_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\combined_value.csv", index=False)

print(" Combined datasets saved successfully!")
print("\n First few rows of Combined Train Data:")
print(combined_train_df.head())

print("\n First few rows of Combined Test Data:")
print(combined_test_df.head())

print("\n First few rows of Combined Value Data:")
print(combined_value_df.head())


# In[12]:


# Defining file paths.
combined_train_path = r"C:\Users\Legion\.jupyter\thesis\combined_train.csv"
combined_test_path = r"C:\Users\Legion\.jupyter\thesis\combined_test.csv"
combined_value_path = r"C:\Users\Legion\.jupyter\thesis\combined_value.csv"

# Load combined datasets
train_df = pd.read_csv(combined_train_path)
test_df = pd.read_csv(combined_test_path)
value_df = pd.read_csv(combined_value_path)


# In[13]:


##data pre processing.##


# In[14]:


#Formatting Timestamps Properly.
def convert_timestamp(ts):
    """Convert timestamp from 'HH-MM-SS-fff' format to datetime."""
    return pd.to_datetime(ts, format="%H-%M-%S-%f")

# Converting timestamp column.
train_df["timestamp"] = train_df["timestamp"].apply(convert_timestamp)
test_df["timestamp"] = test_df["timestamp"].apply(convert_timestamp)
value_df["timestamp"] = value_df["timestamp"].apply(convert_timestamp)


# In[15]:


# Handling Missing Values.
# Checking missing values.
print("\n Missing Values Before Cleaning:")
print(train_df.isnull().sum())

# Filling missing values.
train_df.ffill(inplace=True)
test_df.ffill(inplace=True)
value_df.ffill(inplace=True)


print("\n Missing Values After Cleaning:")
print(train_df.isnull().sum())


# In[16]:


### Validating Data Integrity.


# In[17]:


# Ensuring timestamps are sorted.
train_df = train_df.sort_values(by="timestamp").reset_index(drop=True)
test_df = test_df.sort_values(by="timestamp").reset_index(drop=True)
value_df = value_df.sort_values(by="timestamp").reset_index(drop=True)


# In[18]:


# Checking for duplicate records and remove if necessary.
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)
value_df.drop_duplicates(inplace=True)


# In[19]:


#Saving Cleaned Data.
cleaned_train_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_train.csv"
cleaned_test_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_test.csv"
cleaned_value_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_value.csv"

train_df.to_csv(cleaned_train_path, index=False)
test_df.to_csv(cleaned_test_path, index=False)
value_df.to_csv(cleaned_value_path, index=False)

print("\n Data Preprocessing Completed! Cleaned files saved successfully.")


# In[20]:


# Defining file paths for cleaned train.
cleaned_train_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_train.csv"


# In[21]:


# Loading cleaned traffic data and AQI site locations.
traffic_df = pd.read_csv(cleaned_train_path)
aqi_df = pd.read_csv(aqi_data_path)


# In[22]:


# Converting to GeoDataFrame for spatial operations.
traffic_gdf = gpd.GeoDataFrame(traffic_df, geometry=gpd.points_from_xy(traffic_df.longitude, traffic_df.latitude))
aqi_gdf = gpd.GeoDataFrame(aqi_df, geometry=gpd.points_from_xy(aqi_df.longitude, aqi_df.latitude))


# In[23]:


# Setting Coordinate Reference System (CRS) to WGS 84 (EPSG:4326).
traffic_gdf.set_crs(epsg=4326, inplace=True)
aqi_gdf.set_crs(epsg=4326, inplace=True)


# In[24]:


#Reprojecting to UTM Zone 44N (EPSG:32644) for accurate distance calculations.
traffic_gdf = traffic_gdf.to_crs(epsg=32644)
site_gdf = site_gdf.to_crs(epsg=32644)


# In[25]:


# Performing a spatial join: Find the nearest AQI site for each traffic data point.
joined_df = gpd.sjoin_nearest(traffic_gdf, site_gdf, how="left", distance_col="distance_meters")


# In[26]:


# Converting back to WGS 84.
joined_df = joined_df.to_crs(epsg=4326)


# In[27]:


# Saving the spatially joined dataset.
joined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv", index=False)
# Displaying the first few rows.
print("\n Spatial Join Completed! First few rows of joined dataset:")
print(joined_df.head())


# In[28]:


##### EDA.


# In[29]:


# Loading the spatially joined dataset.
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Creating a base map centered on Telangana.
m = folium.Map(location=[17.5, 78.5], zoom_start=10, tiles="OpenStreetMap")

# Converting latitude & longitude into a format compatible with HeatMap.
heat_data = list(zip(joined_df["latitude"], joined_df["longitude"]))

# Adding heatmap layer.
HeatMap(heat_data, radius=10, blur=15, min_opacity=0.5).add_to(m)

# Saving and display the map.
heatmap_path = r"C:\Users\Legion\.jupyter\thesis\traffic_heatmap_spatially joined dataset.html"
m.save(heatmap_path)

print(f" Traffic Heatmap Saved! Open the file in a browser: {heatmap_path}")


# In[30]:


##traffic and aqi station heatmap locating telegana state boundary.
# Loading the spatially joined dataset.
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Loading the India shapefile.
india_shapefile = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
gdf = gpd.read_file(india_shapefile)

# Filtering to keep only Telangana state.
telangana_gdf = gdf[gdf["st_nm"] == "Telangana"]

# Converting CRS to match Folium.
telangana_gdf = telangana_gdf.to_crs(epsg=4326)

# Creating a base map centered on Telangana.
m = folium.Map(location=[17.5, 78.5], zoom_start=7, tiles="OpenStreetMap")

# Adding Telangana boundary as a layer.
folium.GeoJson(telangana_gdf, name="Telangana Boundary", style_function=lambda x: {
    "color": "black", "weight": 2, "fillOpacity": 0.1
}).add_to(m)

# Converting latitude & longitude into a format compatible with HeatMap.
heat_data = list(zip(joined_df["latitude"], joined_df["longitude"]))

# Adding heatmap layer.
HeatMap(heat_data, radius=10, blur=15, min_opacity=0.5).add_to(m)

# Saving and display the map.
heatmap_path = r"C:\Users\Legion\.jupyter\thesis\traffic_heatmap_aqi station.html"
m.save(heatmap_path)

print(f" Traffic Heatmap Saved! Open the file in a browser: {heatmap_path}")


# In[31]:


## traffic density across telangana.
# Loading the spatially joined dataset.
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Loading the India shapefile.
shapefile_path = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
india_map = gpd.read_file(shapefile_path)

# Filtering to keep only Telangana state.
telangana_map = india_map[india_map["st_nm"] == "Telangana"]

# Creating the plot.
fig, ax = plt.subplots(figsize=(10, 8))

# Plotting Telangana boundary.
telangana_map.plot(ax=ax, edgecolor="black", color="lightgrey", alpha=0.5)

# Overlaying traffic data.
ax.scatter(joined_df["longitude"], joined_df["latitude"], c="red", alpha=0.5, label="Traffic Points")

# Setting labels and title.
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traffic Density Across Telangana with GIS Layer")
plt.legend()
plt.show()


# In[32]:


# Loading the spatially joined dataset.
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Creating a scatter plot of traffic locations.
plt.figure(figsize=(10, 6))
plt.scatter(joined_df["longitude"], joined_df["latitude"], c="red", alpha=0.5, label="Traffic Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traffic Density Across Telangana")
plt.legend()
plt.show()


# In[33]:


# Grouping by AQI site name and calculating the average distance to traffic points.
aqi_site_avg_distance = joined_df.groupby("Site Name")["distance_meters"].mean().reset_index()

# Plotting AQI site pollution trends.
plt.figure(figsize=(12, 6))
sns.barplot(x="distance_meters", y="Site Name", data=aqi_site_avg_distance, hue="Site Name", palette="coolwarm", legend=False)

plt.xlabel("Average Distance to Traffic (meters)")
plt.ylabel("AQI Site")
plt.title("Average Distance of Traffic to AQI Monitoring Sites")
plt.show()


# In[34]:


# Loading AQI dataset.
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Printing column names to verify structure.
print(aqi_df.columns)

# Filtering only 'PM2.5' readings from the indicator column.
pm25_df = aqi_df[aqi_df["indicator"] == "PM2.5"].copy()

# Renaming 'value' column to 'PM2.5' for clarity.
pm25_df = pm25_df.rename(columns={"value": "PM2.5"})

# Displaying the first few rows of PM2.5 data.
print(pm25_df.head())


# In[35]:


# Converting Traffic and PM2.5 data into GeoDataFrames.
traffic_gdf = gpd.GeoDataFrame(joined_df, geometry=gpd.points_from_xy(joined_df["longitude"], joined_df["latitude"]), crs="EPSG:4326")
pm25_gdf = gpd.GeoDataFrame(pm25_df, geometry=gpd.points_from_xy(pm25_df["longitude"], pm25_df["latitude"]), crs="EPSG:4326")

# Converting to UTM for accurate distance calculations.
traffic_gdf = traffic_gdf.to_crs(epsg=32644)
pm25_gdf = pm25_gdf.to_crs(epsg=32644)

# Droping 'index_right' if it exists.
if "index_right" in pm25_gdf.columns:
    pm25_gdf = pm25_gdf.drop(columns=["index_right"])

if "index_right" in traffic_gdf.columns:
    traffic_gdf = traffic_gdf.drop(columns=["index_right"])

# Resetting index to avoid conflicts.
traffic_gdf = traffic_gdf.reset_index(drop=True)
pm25_gdf = pm25_gdf.reset_index(drop=True)

# Finding the nearest PM2.5 measurement for each traffic location.
joined_pm25_df = gpd.sjoin_nearest(traffic_gdf, pm25_gdf, how="left", distance_col="distance_meters")

# Converting back to WGS 84 for visualization.
joined_pm25_df = joined_pm25_df.to_crs(epsg=4326)

# Saving the merged dataset.
joined_pm25_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_pm25_joined.csv", index=False)

print("\n Merged Traffic & PM2.5 Data Saved! First few rows:")
print(joined_pm25_df.head())


# In[36]:


# Computing correlation between traffic distance & PM2.5 levels.
if "distance_meters" in joined_pm25_df.columns and "PM2.5" in joined_pm25_df.columns:
    correlation_value = joined_pm25_df["distance_meters"].corr(joined_pm25_df["PM2.5"])
    print(f" Correlation between Traffic Distance & PM2.5: {correlation_value:.3f}")
else:
    print("Error: Required columns not found! Check column names in joined_pm25_df.")


# In[37]:


# Creating a base map centered on Telangana.
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Adding heatmap layer.
heat_data = list(zip(joined_df["latitude"], joined_df["longitude"]))
HeatMap(heat_data).add_to(m)

# Saving and display the map.
m.save(r"C:\Users\Legion\.jupyter\thesis\traffic_heatmap_base map centered on Telangana.html")
print(" Traffic Heatmap Saved! Open 'traffic_heatmap.html3' to view.")


# In[38]:


## Visualizing the relationship between traffic hotspots and air quality monitoring locations.

# Loading the spatially joined dataset.
df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Creating a base map centered on Telangana.
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Checking if the required columns exist.
if "latitude" in df.columns and "longitude" in df.columns:
    # Adding traffic heatmap.
    heat_data = list(zip(df["latitude"], df["longitude"]))
    HeatMap(heat_data, radius=15).add_to(m)
    
    # Adding AQI sites as markers.
    for _, row in df.iterrows():
        folium.Marker([row["Latitude"], row["Longitude"]],
                      popup=row["Site Name"],
                      icon=folium.Icon(color="red", icon="cloud")).add_to(m)

    # Saving the map.
    m.save(r"C:\Users\Legion\.jupyter\traffic_aqi_density_map.html")
    print(" Traffic Heatmap with AQI Sites Saved! Open 'traffic_aqi_density_map.html' to view.")
else:
    print(" Error: Missing 'latitude' or 'longitude' columns in the dataset!")


# In[39]:


## Method 2: Traffic Density Heatmap with AQI Overlays.


# In[40]:


# Loading the spatially joined dataset.
df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Creating a base map centered on Telangana.
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Checking if the required columns exist.
if "latitude" in df.columns and "longitude" in df.columns:
    # Adding traffic heatmap.
    heat_data = list(zip(df["latitude"], df["longitude"]))
    HeatMap(heat_data, radius=15).add_to(m)
    
    # Adding AQI sites as markers.
    for _, row in df.iterrows():
        folium.Marker([row["Latitude"], row["Longitude"]],
                      popup=row["Site Name"],
                      icon=folium.Icon(color="red", icon="cloud")).add_to(m)

    # Saving the map.
    m.save(r"C:\Users\Legion\.jupyter\traffic_aqi_density_map.html")
    print(" Traffic Heatmap with AQI Sites Saved! Open 'traffic_aqi_density_map.html' to view.")
else:
    print(" Error: Missing 'latitude' or 'longitude' columns in the dataset!")


# In[41]:


# Loading the dataset.
file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_pm25_joined.csv"
df = pd.read_csv(file_path)

# Renaming columns for consistency.
df.rename(columns={
    "latitude_left": "latitude",
    "longitude_left": "longitude"
}, inplace=True)

# Saving the corrected dataset.
df.to_csv(file_path, index=False)
print(" Column names corrected and dataset saved!")


# In[42]:


# Creating a base map centered on Telangana.
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Adding traffic density heatmap.
heat_data = list(zip(df["latitude"], df["longitude"]))
HeatMap(heat_data, radius=15).add_to(m)

# Function to color AQI sites based on PM2.5 levels.
def get_color(pm_value):
    if pm_value < 50:
        return "green"
    elif pm_value < 100:
        return "orange"
    else:
        return "red"

# Adding AQI site markers with PM2.5 values.
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=6,
        color=get_color(row["PM2.5"]),
        fill=True,
        fill_color=get_color(row["PM2.5"]),
        fill_opacity=0.7,
        popup=f"PM2.5 Level: {row['PM2.5']}"
    ).add_to(m)

# Saving the map.
map_path = r"C:\Users\Legion\.jupyter\thesis\traffic_pm25_heatmap.html"
m.save(map_path)

print(f" Traffic Density Heatmap with PM2.5 Overlays Saved! Open '{map_path}' to view.")


# In[43]:


# Loading AQI monitoring site data.
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\site.csv")

# Creating a base map.
aqi_map = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Function to color AQI sites based on pollution level.
def get_color(aqi_value):
    if aqi_value < 50:
        return "green"
    elif aqi_value < 100:
        return "orange"
    else:
        return "red"

# Adding AQI monitoring sites to the map.
for _, row in aqi_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,
        color=get_color(100),  # Replace 100 with real AQI values if available
        fill=True,
        fill_color=get_color(100),
        fill_opacity=0.7,
        popup=f"AQI Site: {row['Site Name']}"
    ).add_to(aqi_map)

# Saving and display the AQI map.
aqi_map.save(r"C:\Users\Legion\.jupyter\thesis\aqi_map_station.html")
print("AQI Map Saved! Open 'aqi_map.html' to view.")


# In[44]:


# Creating base map.
combined_map = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Adding traffic heatmap.
HeatMap(heat_data).add_to(combined_map)

# Adding AQI monitoring sites with color coding.
for _, row in aqi_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,
        color=get_color(100),  # Replace with real AQI value
        fill=True,
        fill_color=get_color(100),
        fill_opacity=0.7,
        popup=f"AQI Site: {row['Site Name']}"
    ).add_to(combined_map)

# Saveing and display combined map.
combined_map.save(r"C:\Users\Legion\.jupyter\thesis\combined_traffic_aqi_map.html")
print(" Combined Traffic & AQI Map Saved! Open 'combined_traffic_aqi_map2.html' to view.")


# In[45]:


##AQI Heatmap


# In[46]:


# Loading AQI data.
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Creating a base map centered on Telangana.
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Adding AQI heatmap.
heat_data = list(zip(aqi_df["latitude"], aqi_df["longitude"]))
HeatMap(heat_data).add_to(m)

# Saveing & Display the map.
m.save(r"C:\Users\Legion\.jupyter\thesis\aqi_heatmap.html")
print(" AQI Heatmap Saved! Open 'aqi_heatmap.html11' to view.")


# In[47]:


##Hourly AQI Trend.


# In[48]:


# Loading the AQI dataset.
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Converting 'date' and 'time' into proper datetime format.
aqi_df["datetime"] = pd.to_datetime(aqi_df["date"] + " " + aqi_df["time"], format="%d/%m/%Y %H:%M")

# Ensure 'indicator' column is properly formatted.
aqi_df["indicator"] = aqi_df["indicator"].astype(str)

# Filtering data for a specific date July 11, 2019).
aqi_day = aqi_df[aqi_df["datetime"].dt.date == pd.to_datetime("2019-07-11").date()]

# Plotting AQI trend by hour.
plt.figure(figsize=(12, 6))
sns.lineplot(data=aqi_day, x="datetime", y="value", hue="indicator", marker="o", palette="tab10")

plt.xlabel("Time of Day")
plt.ylabel("AQI Value")
plt.title("Hourly AQI Trends on July 11, 2019")
plt.xticks(rotation=45)
plt.legend(title="Pollutant Type", loc="upper left", bbox_to_anchor=(1, 1))
plt.show()


# In[49]:


# Plotting pollutant distribution for July 11, 2019.
plt.figure(figsize=(10, 5))
sns.boxplot(data=aqi_day, x="indicator", y="value")
plt.xlabel("Pollutant Type")
plt.ylabel("AQI Value")
plt.title("Distribution of Different Pollutants on July 11, 2019")
plt.xticks(rotation=45)
plt.show()


# In[50]:


# Averaging AQI per location.
aqi_location_avg = aqi_day.groupby("location")["value"].mean().reset_index()

# Plotting AQI by location.
plt.figure(figsize=(12, 6))
sns.barplot(data=aqi_location_avg, x="value", y="location", palette="coolwarm")
plt.xlabel("Average AQI")
plt.ylabel("Monitoring Site")
plt.title("Average AQI Levels by Location on July 11, 2019")
plt.show()


# In[51]:


##Merging Traffic & AQI Data on Time.


# In[52]:


# Loading traffic data (cleaned).
traffic_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\cleaned_train.csv")

# Loading AQI data.
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Converting timestamps to datetime format.
traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
aqi_df["datetime"] = pd.to_datetime(aqi_df["date"] + " " + aqi_df["time"], format="%d/%m/%Y %H:%M")

# Extracting hour from timestamps for aggregation.
traffic_df["hour"] = traffic_df["timestamp"].dt.hour
aqi_df["hour"] = aqi_df["datetime"].dt.hour

# Aggregating traffic data (count of traffic records per hour).
traffic_hourly = traffic_df.groupby("hour").size().reset_index(name="traffic_count")

# Aggregating AQI data (average AQI per hour).
aqi_hourly = aqi_df.groupby(["hour", "indicator"])["value"].mean().reset_index()

# Merging traffic & AQI data on "hour".
combined_df = pd.merge(aqi_hourly, traffic_hourly, on="hour", how="left")

# Saving the merged dataset for verification.
combined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_merged.csv", index=False)

# Displaying the first few rows for preview.
print("\n Traffic & AQI Merged Data (First 5 rows):")
print(combined_df.head())


# In[53]:


## Visualizing Traffic vs. AQI Trends.


# In[56]:


plt.figure(figsize=(12, 6))

# AQI trend.
sns.lineplot(data=combined_df, x="hour", y="value", hue="indicator", marker="o")

# Traffic trend.
sns.lineplot(data=combined_df, x="hour", y="traffic_count", color="red", marker="s")

# Labels and title.
plt.xlabel("Hour of Day")
plt.ylabel("AQI Value / Traffic Count")
plt.title("Traffic vs. AQI Trends Throughout the Day")

# Showing legend automatically from hue.
plt.legend()
plt.show()


# In[57]:


##Correlation Between Traffic & AQI


# In[58]:


# Computing correlation between traffic and AQI values.
correlation_matrix = combined_df.pivot(index="hour", columns="indicator", values="value").corrwith(combined_df["traffic_count"])
print("\n Correlation Between Traffic & AQI Indicators:")
print(correlation_matrix)


# In[59]:


# Loading datasets
traffic_aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")  # Spatially joined dataset
aqi_data = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")  # AQI readings dataset

# Converting timestamps to datetime format
traffic_aqi_df["timestamp"] = pd.to_datetime(traffic_aqi_df["timestamp"])
aqi_data["datetime"] = pd.to_datetime(aqi_data["date"] + " " + aqi_data["time"], format="%d/%m/%Y %H:%M")

# Extracting hour for merging
traffic_aqi_df["hour"] = traffic_aqi_df["timestamp"].dt.hour
aqi_data["hour"] = aqi_data["datetime"].dt.hour

# Merging AQI data into traffic dataset based on hour and nearest location
merged_df = pd.merge(traffic_aqi_df, aqi_data, on="hour", how="left")

# Saving merged dataset
merged_file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final.csv"
merged_df.to_csv(merged_file_path, index=False)

print(" AQI Data Successfully Merged into Traffic Dataset! Ready for analysis.")


# In[60]:


# Load the merged dataset
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final.csv")

# Fixing timestamp format: Strip extra microseconds and convert to datetime.
combined_df["timestamp"] = combined_df["timestamp"].str.split(".").str[0]  # Remove microseconds if present
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], errors="coerce")  # Convert properly

# Extracting hour for time-based analysis.
combined_df["hour"] = combined_df["timestamp"].dt.hour

# Saving the corrected dataset.
combined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv", index=False)

print(" Timestamp Issue Fixed! Dataset saved as 'traffic_aqi_final_fixed.csv'.")


# In[61]:


# Loading the dataset.
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final.csv")

# Ensuring 'timestamp' column is treated as string.
combined_df["timestamp"] = combined_df["timestamp"].astype(str)

# Removing any extra spaces or unexpected characters.
combined_df["timestamp"] = combined_df["timestamp"].str.strip()

# Trying different formats and clean microseconds if necessary.
def clean_timestamp(ts):
    try:
        # Removing microseconds if present.
        if "." in ts:
            ts = ts.split(".")[0]
        # Converting to datetime.
        return pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    except:
        return pd.NaT  # Assigning NaT (Not a Time) if parsing fails.

# Applying timestamp cleaning function.
combined_df["timestamp"] = combined_df["timestamp"].apply(clean_timestamp)

# Dropping any rows with NaT timestamps.
combined_df = combined_df.dropna(subset=["timestamp"])

# Extracting hour from timestamp for time-based analysis.
combined_df["hour"] = combined_df["timestamp"].dt.hour

# Saving the cleaned dataset.
fixed_file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv"
combined_df.to_csv(fixed_file_path, index=False)

print(" Timestamp Issue Fixed! Cleaned dataset saved as 'traffic_aqi_final_fixed.csv'.")


# In[62]:


# Loading the dataset.
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv")

# Converting timestamp to datetime format.
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

# Counting traffic events per hour.
traffic_count_per_hour = combined_df.groupby("hour").size().reset_index(name="traffic_count")

# Merging back into the dataset.
combined_df = pd.merge(combined_df, traffic_count_per_hour, on="hour", how="left")

# Saving the updated dataset.
combined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv", index=False)

print(" Traffic count successfully added to the dataset!")


# In[64]:


# Reloading the fixed dataset.
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv")

# Converting timestamp to datetime format.
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

# Aggregating AQI values per hour.
aggregated_df = combined_df.groupby(["hour", "indicator"], as_index=False).agg({"value": "mean", "traffic_count": "sum"})

# Pivot after aggregation.
pivot_df = aggregated_df.pivot(index="hour", columns="indicator", values="value")

# Computing correlation between traffic count and AQI values.
correlation_matrix = pivot_df.corrwith(aggregated_df.groupby("hour")["traffic_count"].sum())

# Saving correlation results to CSV.
correlation_matrix.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_correlation.csv")

# Printing correlation results.
print("\n Traffic & AQI Correlation Results:")
print(correlation_matrix)

# Plotting AQI & Traffic trends over time.
plt.figure(figsize=(12, 6))

# AQI trend.
sns.lineplot(data=combined_df, x="hour", y="value", hue="indicator", marker="o")

# Traffic trend.
sns.lineplot(data=combined_df, x="hour", y="traffic_count", color="red", marker="s")

plt.xlabel("Hour of Day")
plt.ylabel("AQI Value / Traffic Count")
plt.title("Traffic vs. AQI Trends Throughout the Day")
plt.legend()
plt.show()


# In[65]:


##Correlation Analysis , Fix: Normalize Traffic & AQI Data


# In[66]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Initializing MinMaxScaler.
scaler = MinMaxScaler()

# Scaling 'traffic_count' to range 0-1.
aggregated_df["traffic_count_scaled"] = scaler.fit_transform(aggregated_df[["traffic_count"]])

# Plotting AQI & Traffic trends over time.
plt.figure(figsize=(14, 6))

# AQI trend.
sns.lineplot(data=aggregated_df, x="hour", y="value", hue="indicator", marker="o")

# Traffic trend (scaled)
sns.lineplot(data=aggregated_df, x="hour", y="traffic_count_scaled", color="red", marker="s", label="Traffic Count (Scaled)")

plt.xlabel("Hour of Day")
plt.ylabel("AQI Value / Scaled Traffic Count")
plt.title("Traffic vs. AQI Trends Throughout the Day")

# Moving legend outside the plot.
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Adjust y-axis limits for better visibility.
plt.ylim(0, max(aggregated_df["value"].max(), 1))  # Since traffic is scaled, upper limit is 1

plt.show()


# In[67]:


##ML modeling.


# In[68]:


# Loading dataset.
file_path = r"C:\Users\Legion\.jupyter\traffic_aqi_final_fixed.csv"
df = pd.read_csv(file_path)

# Converting timestamp to datetime format.
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Printing dataset info.
print(" Dataset Info:")
print(df.info())

# Displaying first few rows.
print("\n Sample Data:")
print(df.head())


# In[70]:


# Checking for missing values.
print("\n❗ Missing Values Before Handling:")
print(df.isnull().sum())

# Filling missing values using forward fill for time-series data.
df.ffill(inplace=True)  # ✅ Correct way to apply forward fill

# Verifying missing values are handled.
print("\n✅ Missing Values After Handling:")
print(df.isnull().sum())


# In[71]:


# Loading the dataset.
file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv"
df = pd.read_csv(file_path)

# Converting timestamp to datetime format.
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extracting hour for time-based modeling.
df["hour"] = df["timestamp"].dt.hour

# Pivot AQI data.
df_pivot = df.pivot_table(index=["timestamp", "hour", "traffic_count"], 
                          columns="indicator", values="value").reset_index()

# Saving the pivoted dataset.
df_pivot.to_csv(r"C:\Users\Legion\.jupyter\thesis\pivoted_data.csv", index=False)

print(" AQI Data Successfully Pivoted! Saved as 'pivoted_data.csv'.")


# In[72]:


# Reloading the pivoted dataset.
df_ml = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\pivoted_data.csv")

# Selecting relevant features.
selected_features = ["hour", "traffic_count", "NO2", "PM2.5", "Ozone", "CO", "SO2"]

# Filtering dataset.
df_ml = df_ml[selected_features]

# Saving processed dataset.
df_ml.to_csv(r"C:\Users\Legion\.jupyter\thesis\processed_data.csv", index=False)

print(" Feature Selection Completed! Processed data saved as 'processed_data.csv'.")


# In[73]:


# Loading dataset.
df_ml = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\processed_data.csv")

# Initializing scaler.
scaler = MinMaxScaler()

# Scaling data.
df_scaled = pd.DataFrame(scaler.fit_transform(df_ml), columns=df_ml.columns)

# Saving scaled dataset.
df_scaled.to_csv(r"C:\Users\Legion\.jupyter\thesis\scaled_data.csv", index=False)

print(" Data Scaling Completed! Saved as 'scaled_data.csv'.")


# In[74]:


# train and evaluate random forest model.


# In[75]:


# Loading the processed dataset.
df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\scaled_data.csv")

# Defining input features (X) and target variable (y).
X = df.drop(columns=["PM2.5"])  # Predicting PM2.5 levels
y = df["PM2.5"]

# Splitting data into train & test sets (80% train, 20% test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Random Forest Model.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions.
y_pred = rf_model.predict(X_test)

# Evaluating model performance.
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n Random Forest Model Evaluation:")
print(f" Mean Absolute Error (MAE): {mae:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f" R² Score: {r2:.2f}")

# Feature Importance.
importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)

print("\n Feature Importance Ranking:")
print(importances)

# Saving model predictions.
predictions_df = pd.DataFrame({"Actual_PM2.5": y_test, "Predicted_PM2.5": y_pred})
predictions_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\rf_predictions.csv", index=False)

print("\n Predictions saved as 'rf_predictions.csv'")


# In[77]:


import joblib  # Importing joblib for saving models.

# Saveing the trained Random Forest model.
model_path = r"C:\Users\Legion\.jupyter\thesis\rf_model.pkl"
joblib.dump(rf_model, model_path)

print(f"\n Random Forest Model saved as '{model_path}'")


# In[76]:


# train and evaluate LSTM


# In[79]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Defining input shape (assuming X_train is already shaped properly).
input_shape = (X_train.shape[1], 1)  # Adjusting based on data preprocessing.

#  Corrected Model
model = Sequential([
    Input(shape=input_shape),  # Defining input layer explicitly.
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer for regression.
])

# Compiling the model.
model.compile(optimizer="adam", loss="mse")

# Printing model summary.
model.summary()


# In[80]:


# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)


# In[83]:


# Saving the trained LSTM model
lstm_model_path = r"C:\Users\Legion\.jupyter\thesis\lstm_model.keras"
model.save(r"C:\Users\Legion\.jupyter\thesis\lstm_model.keras")

print(f"\n LSTM Model saved as '{lstm_model_path}'")


# In[84]:


# Saving the data scaler (useful for re-scaling future predictions).
scaler_path = r"C:\Users\Legion\.jupyter\thesis\scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"\n Data scaler saved as '{scaler_path}'")


# In[86]:


import tensorflow as tf

# Loading the saved LSTM model.
lstm_model = tf.keras.models.load_model(r"C:\Users\Legion\.jupyter\thesis\lstm_model.keras")

# Making predictions.
y_pred_lstm = lstm_model.predict(X_test)

# Saving. predictions
lstm_predictions_df = pd.DataFrame({"Actual_PM2.5": y_test, "Predicted_PM2.5": y_pred_lstm.flatten()})
lstm_predictions_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\lstm_predictions.csv", index=False)

print(" LSTM Predictions Saved Successfully at 'lstm_predictions.csv'!")


# In[87]:


import matplotlib.pyplot as plt

# Plotting training and validation loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LSTM Training Progress (Loss Curve)")
plt.legend()
plt.grid()
plt.show()


# In[88]:


# Computing Evaluation Metrics.
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
r2_lstm = r2_score(y_test, y_pred_lstm)

# Printing the results.
print("\n🔍 LSTM Model Evaluation:")
print(f"📌 Mean Absolute Error (MAE): {mae_lstm:.2f}")
print(f"📌 Root Mean Squared Error (RMSE): {rmse_lstm:.2f}")
print(f"📌 R² Score: {r2_lstm:.2f}")


# In[89]:


#Compare Performance (RF vs. LSTM).


# In[90]:


# Loading Predictions.
rf_preds = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\rf_predictions.csv") 
lstm_preds = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\lstm_predictions.csv") 

#Computing Performance Metrics.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {"Random Forest": rf_preds, "LSTM": lstm_preds}
results = {}

for model_name, df in models.items():
    mae = mean_absolute_error(df["Actual_PM2.5"], df["Predicted_PM2.5"])
    rmse = np.sqrt(mean_squared_error(df["Actual_PM2.5"], df["Predicted_PM2.5"]))
    r2 = r2_score(df["Actual_PM2.5"], df["Predicted_PM2.5"])
    
    results[model_name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

# Converting results to DataFrame.
results_df = pd.DataFrame(results).T

# Saving results.
results_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\model_comparison.csv")

# Printing comparison table.
print("\n Model Performance Comparison:")
print(results_df)


# In[91]:


#Visualize Predictions


# In[92]:


# Ploting Actual vs Predicted AQI (PM2.5)
plt.figure(figsize=(12, 6))

# Random Forest Predictions
sns.lineplot(data=rf_preds[:100], x=rf_preds.index[:100], y="Actual_PM2.5", label="Actual PM2.5", linestyle="dashed")
sns.lineplot(data=rf_preds[:100], x=rf_preds.index[:100], y="Predicted_PM2.5", label="RF Predicted PM2.5")

# LSTM Predictions
sns.lineplot(data=lstm_preds[:100], x=lstm_preds.index[:100], y="Predicted_PM2.5", label="LSTM Predicted PM2.5")

plt.xlabel("Time Steps")
plt.ylabel("PM2.5 AQI Level")
plt.title("Actual vs. Predicted AQI (PM2.5) - RF vs LSTM")
plt.legend()
plt.show()


# In[96]:


# Ensuring Random Forest Predictions exist.
y_pred_rf = rf_model.predict(X_test)

#  Comparing LSTM vs. Random Forest Predictions.
plt.figure(figsize=(12, 6))

plt.plot(y_test.values, label="Actual PM2.5", linestyle="-", marker="o", color="blue")

plt.plot(y_pred_rf, label="Predicted PM2.5 (RF)", linestyle="--", marker="s", color="green")

plt.plot(y_pred_lstm.flatten(), label="Predicted PM2.5 (LSTM)", linestyle="--", marker="^", color="red")

# Formatting the plot.
plt.xlabel("Test Sample Index")
plt.ylabel("PM2.5 Concentration")
plt.title(" Model Comparison: LSTM vs. Random Forest Predictions")
plt.legend()
plt.grid(True)

plt.show()


# In[105]:


# Scatter Plot: Actual vs. Predicted PM2.5 (Both Models).
plt.figure(figsize=(12, 6))

# Random Forest scatter plot.
plt.scatter(y_test, y_pred_rf, label="Random Forest", color="green", alpha=0.6)

# LSTM scatter plot.
plt.scatter(y_test, y_pred_lstm.flatten(), label="LSTM", color="red", alpha=0.6)

# Perfect prediction reference line.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

# Formatting.
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("📊 Actual vs. Predicted PM2.5 (LSTM & Random Forest)")
plt.legend()
plt.grid(True)

plt.show()


# In[103]:


# Computeing Residuals (Errors).
rf_residuals = y_test - y_pred_rf
lstm_residuals = y_test - y_pred_lstm.flatten()

# Histogram of Residuals.
plt.figure(figsize=(12, 6))

# Random Forest Residuals.
plt.hist(rf_residuals, bins=30, alpha=0.6, label="RF Residuals", color="green")

# LSTM Residuals.
plt.hist(lstm_residuals, bins=30, alpha=0.6, label="LSTM Residuals", color="red")

# Formatting.
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("📉 Residual Distribution: LSTM vs. Random Forest")
plt.legend()
plt.grid(True)
plt.show()


# In[101]:


# Feature Importance - Random Forest.(as no for LSTM)
importances = pd.DataFrame({'Feature': X_test.columns, 'Importance': rf_model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)

# Plotting feature importance.
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'], importances['Importance'], color='purple')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Model Comparison: LSTM vs. Random Forest Predictions")
plt.gca().invert_yaxis()  
plt.show()
print("\n Feature Importance Ranking (Random Forest):")
print(importances)


# In[106]:


##Hyperparameter Tuning.


# In[107]:


# RANDOM FOREST


# In[108]:


from sklearn.model_selection import GridSearchCV

# Defining hyperparameter grid.
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initializing Random Forest model.
rf_model = RandomForestRegressor(random_state=42)

# Performing Grid Search.
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring="r2")

# Fit model to training data.
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search.
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Training the model with the best parameters.
best_rf_model = grid_search.best_estimator_

# Making predictions.
y_pred_rf_tuned = best_rf_model.predict(X_test)

# Evaluating the tuned model.
mae_rf_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
rmse_rf_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
r2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)

# Printing evaluation metrics.
print("\n Tuned Random Forest Model Evaluation:")
print(f" Mean Absolute Error (MAE): {mae_rf_tuned:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse_rf_tuned:.2f}")
print(f" R² Score: {r2_rf_tuned:.2f}")


# In[109]:


import joblib

# Saveing the tuned Random Forest model.
rf_tuned_model_path = r"C:\Users\Legion\.jupyter\thesis\rf_tuned_model.pkl"
joblib.dump(best_rf_model, rf_tuned_model_path)

print(f" Tuned Random Forest Model saved as '{rf_tuned_model_path}'")


# In[112]:


import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Defining a function to build the LSTM model.
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int("units_1", min_value=32, max_value=128, step=32),
                   return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float("dropout_1", min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(LSTM(hp.Int("units_2", min_value=32, max_value=128, step=32),
                   return_sequences=False))
    model.add(Dropout(hp.Float("dropout_2", min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(Dense(hp.Int("dense_units", min_value=16, max_value=64, step=16), activation="relu"))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", [0.001, 0.0005, 0.0001])),
                  loss="mse")
    
    return model

# Initializing Keras Tuner.
tuner = kt.RandomSearch(
    build_lstm_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    directory="lstm_tuning",
    project_name="lstm_pm25"
)

# Performing Hyperparameter Tuning.
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Getting the best hyperparameters.
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps.values}")

# Building the best model and train it.
best_lstm_model = tuner.hypermodel.build(best_hps)
history = best_lstm_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)

# Making predictions with tuned LSTM.
y_pred_lstm_tuned = best_lstm_model.predict(X_test)

# Evaluating the tuned LSTM model.
mae_lstm_tuned = mean_absolute_error(y_test, y_pred_lstm_tuned)
rmse_lstm_tuned = np.sqrt(mean_squared_error(y_test, y_pred_lstm_tuned))
r2_lstm_tuned = r2_score(y_test, y_pred_lstm_tuned)

# Printing evaluation metrics.
print("\n Tuned LSTM Model Evaluation:")
print(f" Mean Absolute Error (MAE): {mae_lstm_tuned:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse_lstm_tuned:.2f}")
print(f" R² Score: {r2_lstm_tuned:.2f}")


# In[113]:


# Saveing the tuned LSTM model.
lstm_tuned_model_path = r"C:\Users\Legion\.jupyter\thesis\lstm_tuned_model.keras"
best_lstm_model.save(lstm_tuned_model_path)

print(f" Tuned LSTM Model saved as '{lstm_tuned_model_path}'")


# In[ ]:


## Visualizing Hyperparameter Tuning for Random Forest.


# In[116]:


# Converting GridSearchCV results to DataFrame,
cv_results = pd.DataFrame(grid_search.cv_results_)

# Extractting relevant columns.
cv_results = cv_results[['param_n_estimators', 'param_max_depth', 'mean_test_score']]
cv_results = cv_results.pivot_table(index='param_n_estimators', columns='param_max_depth', values='mean_test_score', aggfunc='mean')


# Plotting heatmap.
plt.figure(figsize=(8, 6))
sns.heatmap(cv_results, annot=True, cmap="coolwarm")
plt.title(" Hyperparameter Tuning (Random Forest)")
plt.xlabel("Max Depth")
plt.ylabel("Number of Estimators")
plt.show()


# In[ ]:


## Visualizing Hyperparameter Tuning for LSTM (Keras Tuner).


# In[117]:


# Extracting best hyperparameters.
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n Best Hyperparameters Found:")
for hp in best_hps.values:
    print(f"{hp}: {best_hps.get(hp)}")


# In[118]:


# Extracting tuning history.
tuner_results = tuner.oracle.trials

# Converting to DataFrame for visualization.
trials = []
for trial_id, trial in tuner_results.items():
    trials.append({
        "Trial": trial_id,
        "Units 1": trial.hyperparameters.values["units_1"],
        "Units 2": trial.hyperparameters.values["units_2"],
        "Dropout 1": trial.hyperparameters.values["dropout_1"],
        "Dropout 2": trial.hyperparameters.values["dropout_2"],
        "Dense Units": trial.hyperparameters.values["dense_units"],
        "Learning Rate": trial.hyperparameters.values["learning_rate"],
        "Final Loss": trial.score
    })

df_tuning = pd.DataFrame(trials)

# Plotting results.
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_tuning, x="Trial", y="Final Loss", marker="o")
plt.title("Loss Across Hyperparameter Tuning Trials (LSTM)")
plt.xlabel("Trial Number")
plt.ylabel("Final Validation Loss")
plt.grid(True)
plt.show()


# In[ ]:





# In[119]:


# Plotting predictions vs actual values for Random Forest
plt.figure(figsize=(10,6))
plt.plot(y_test, label="Actual Values", color='blue', linestyle='--')
plt.plot(y_pred_rf, label="Predicted Values (Random Forest)", color='red')
plt.legend()
plt.xlabel("Data Points")
plt.ylabel("AQI Values")
plt.title("Predicted vs Actual AQI Values (Random Forest)")
plt.show()


# In[120]:


plt.figure(figsize=(10, 6))

# Scatter plot for actual vs predicted values.
plt.scatter(y_test, y_pred_rf, color="red", alpha=0.5, label="Predicted vs Actual")

# Plotting a perfect prediction reference line (y = x).
min_val = min(min(y_test), min(y_pred_rf))
max_val = max(max(y_test), max(y_pred_rf))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="blue", label="Perfect Prediction (y=x)")

plt.xlabel("Actual AQI Values")
plt.ylabel("Predicted AQI Values")
plt.title("Predicted vs. Actual AQI Values (Random Forest)")
plt.legend()
plt.grid(True)
plt.show()


# In[124]:


plt.figure(figsize=(10,6))

# Scatter plot for actual vs predicted values.
plt.scatter(y_test, y_pred_lstm, color="red", alpha=0.5, label="Predicted vs Actual")

# Perfect prediction reference line (y = x).
min_val = min(min(y_test), min(y_pred_lstm))
max_val = max(max(y_test), max(y_pred_lstm))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="blue", label="Perfect Prediction (y=x)")

plt.xlabel("Actual AQI Values")
plt.ylabel("Predicted AQI Values")
plt.title("Predicted vs. Actual AQI Values (LSTM)")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Plotting heatmap for Random Forest Hyperparameter Tuning.


# In[128]:


# Extracting best model scores.
best_rf_score = grid_search.best_score_  
best_lstm_score = -min(df_tuning["Final_Loss"])  

# Data for comparison.
comparison_df = pd.DataFrame({
    "Model": ["Random Forest", "LSTM"],
    "Best Score": [best_rf_score, best_lstm_score]
})

# Plotting comparison.
plt.figure(figsize=(8, 5))
sns.barplot(data=comparison_df, x="Model", y="Best Score", hue="Model", palette="viridis", legend=False)

plt.title("Best Model Scores: Random Forest vs. LSTM")  
plt.ylabel("Best Performance Score")
plt.show()


# In[130]:


# Plotting residuals for Random Forest.
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, y_test - y_pred_rf, color='blue', label='Random Forest')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted AQI')
plt.ylabel('Residuals')
plt.title('Residual Plot (Random Forest)')
plt.legend()
plt.show()

# Converting y_test to NumPy before flattening.
y_test_flat = y_test.to_numpy().flatten()
y_pred_lstm_flat = y_pred_lstm.flatten()

# Plotting residuals for LSTM
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lstm_flat, y_test_flat - y_pred_lstm_flat, color='green', label='LSTM')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted AQI')
plt.ylabel('Residuals')
plt.title('Residual Plot (LSTM)')
plt.legend()
plt.show()


# In[138]:


# Ensuring MSE is calculated for both models.
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

#  Ensuring R² Score is also calculated.
r2_rf = r2_score(y_test, y_pred_rf)
r2_lstm = r2_score(y_test, y_pred_lstm)

# Labels for models.
labels = ['Random Forest', 'LSTM']

#  Plotting MSE for both models.
plt.figure(figsize=(10, 6))
plt.bar(labels, [mse_rf, mse_lstm], color=['blue', 'green'])
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Model Performance Comparison (MSE)')
plt.show()

# Ploting R² for both models.
plt.figure(figsize=(10, 6))
plt.bar(labels, [r2_rf, r2_lstm], color=['blue', 'green'])
plt.ylabel('R² Score')
plt.title('Model Performance Comparison (R²)')
plt.show()


# In[146]:


# Defining LSTM model with correct input shape (1, features)
model = Sequential([
    Input(shape=(1, X_train.shape[1])),  # (time steps=1, features=6)
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

#  Compiling the model.
model.compile(optimizer="adam", loss="mse")

# Training the model.
history = model.fit(
    X_train_reshaped, y_train,
    epochs=100, batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)

# Plotting Learning Curves.
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title('Learning Curves (LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[150]:


# Ensuring y_test and y_pred_lstm are 1D arrays.
y_test_flat = np.ravel(y_test) 
y_pred_lstm_flat = np.ravel(y_pred_lstm)

# Calculating cumulative error for LSTM.
cumulative_error_lstm = np.cumsum(np.abs(y_test_flat - y_pred_lstm_flat))

# Plotting cumulative error for LSTM.
plt.figure(figsize=(10, 6))
plt.plot(cumulative_error_lstm, label='Cumulative Error (LSTM)', color='green')
plt.xlabel('Data Points')
plt.ylabel('Cumulative Error')
plt.title('Cumulative Error Plot (LSTM)')
plt.legend()
plt.show()

## Ensure 1D arrays for RANDOM FOREST.
y_test_flat = np.ravel(y_test)  
y_pred_rf_flat = np.ravel(y_pred_rf)  

# Calculate cumulative error for Random Forest
cumulative_error_rf = np.cumsum(np.abs(y_test_flat - y_pred_rf_flat))

# Plotting cumulative error correctly.
plt.figure(figsize=(10, 6))
plt.plot(range(len(cumulative_error_rf)), cumulative_error_rf, label='Cumulative Error (Random Forest)', color='blue')
plt.xlabel('Data Points')
plt.ylabel('Cumulative Error')
plt.title('Cumulative Error Plot (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




