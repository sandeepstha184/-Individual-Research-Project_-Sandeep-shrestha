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


#Load GIS Shapefile of India
shapefile_path = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
india_map = gpd.read_file(shapefile_path)


# In[3]:


india_map = gpd.read_file(shapefile_path)
india_map.plot()


# In[4]:


plt.show()
india_map.head() 


# In[5]:


# Customizing the plot
india_map.plot(edgecolor='black', color='lightblue', figsize=(10, 10))

plt.title('Map of Indian States')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[6]:


print(india_map['st_nm'])


# In[7]:


# Filtering to show only a specific state
Telangana_map = india_map[india_map['st_nm'] == 'Telangana']

# Ploting the filtered map
Telangana_map.plot(edgecolor='black', color='green')
plt.title('Map of Telangana')
plt.show()


# In[8]:


# Load AQI site CSV file
site_data_path = r"C:\Users\Legion\.jupyter\thesis\site.csv" 
site_df = pd.read_csv(site_data_path)

# Display first few rows to verify correct loading
print(site_df.head())


# In[9]:


# Load AQI site CSV file
aqi_data_path = r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv" 
aqi_df = pd.read_csv(aqi_data_path)

# Display first few rows to verify correct loading
print(aqi_df.head())


# In[10]:


# Convert site data into a GeoDataFrame
site_gdf = gpd.GeoDataFrame(site_df, geometry=gpd.points_from_xy(site_df.Longitude, site_df.Latitude), crs="EPSG:4326")

# Load India shapefile and filter Telangana
shapefile_path = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
india_map = gpd.read_file(shapefile_path)
telangana_map = india_map[india_map["st_nm"] == "Telangana"]

# Plot Telangana Map
fig, ax = plt.subplots(figsize=(10, 8))
telangana_map.plot(ax=ax, color="lightgrey", edgecolor="black")


# Plot AQI Site Locations
site_gdf.plot(ax=ax, markersize=20, color="blue", alpha=0.8, edgecolor="black", marker="o", label="AQI Sites")

# Customize the map
plt.title("Air Quality Index Monitoring Sites in Telangana")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()


# In[11]:


# Define primary file paths (loading from all subdirectories)
primary_folder = r"C:\Users\Legion\.jupyter\thesis\primary"
primary_train_files = glob.glob(primary_folder + r"\d*\train.csv")
primary_test_files = glob.glob(primary_folder + r"\d*\test.csv")
primary_value_files = glob.glob(primary_folder + r"\d*\value.csv")

# Define synthetic file paths (with correct filenames)
synthetic_folder = r"C:\Users\Legion\.jupyter\thesis\synthetic"
synthetic_train_path = synthetic_folder + r"\synthetic_train_data.csv"
synthetic_test_path = synthetic_folder + r"\synthetic_test_data.csv"
synthetic_value_path = synthetic_folder + r"\synthetic_value_data.csv"

# Load primary datasets (from multiple folders)
primary_train_df = pd.concat([pd.read_csv(f) for f in primary_train_files], ignore_index=True)
primary_test_df = pd.concat([pd.read_csv(f) for f in primary_test_files], ignore_index=True)
primary_value_df = pd.concat([pd.read_csv(f) for f in primary_value_files], ignore_index=True)

# Load synthetic datasets with correct filenames
synthetic_train_df = pd.read_csv(synthetic_train_path)
synthetic_test_df = pd.read_csv(synthetic_test_path)
synthetic_value_df = pd.read_csv(synthetic_value_path)

# Ensure column consistency
assert list(primary_train_df.columns) == list(synthetic_train_df.columns), "Train Data Columns Do Not Match!"
assert list(primary_test_df.columns) == list(synthetic_test_df.columns), "Test Data Columns Do Not Match!"
assert list(primary_value_df.columns) == list(synthetic_value_df.columns), "Value Data Columns Do Not Match!"

# Combine primary and synthetic datasets
combined_train_df = pd.concat([primary_train_df, synthetic_train_df], ignore_index=True)
combined_test_df = pd.concat([primary_test_df, synthetic_test_df], ignore_index=True)
combined_value_df = pd.concat([primary_value_df, synthetic_value_df], ignore_index=True)

# Ensure timestamps are sorted properly
combined_train_df = combined_train_df.sort_values(by="timestamp").reset_index(drop=True)
combined_test_df = combined_test_df.sort_values(by="timestamp").reset_index(drop=True)
combined_value_df = combined_value_df.sort_values(by="timestamp").reset_index(drop=True)

# Save and display first few rows for verification
combined_train_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\combined_train.csv", index=False)
combined_test_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\combined_test.csv", index=False)
combined_value_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\combined_value.csv", index=False)

print("✅ Combined datasets saved successfully!")
print("\n📂 First few rows of Combined Train Data:")
print(combined_train_df.head())

print("\n📂 First few rows of Combined Test Data:")
print(combined_test_df.head())

print("\n📂 First few rows of Combined Value Data:")
print(combined_value_df.head())


# In[12]:


# Define file paths
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


#Format Timestamps Properly
def convert_timestamp(ts):
    """Convert timestamp from 'HH-MM-SS-fff' format to datetime."""
    return pd.to_datetime(ts, format="%H-%M-%S-%f")

# Convert timestamp column
train_df["timestamp"] = train_df["timestamp"].apply(convert_timestamp)
test_df["timestamp"] = test_df["timestamp"].apply(convert_timestamp)
value_df["timestamp"] = value_df["timestamp"].apply(convert_timestamp)


# In[15]:


# Handle Missing Values
# Check missing values
print("\n🔍 Missing Values Before Cleaning:")
print(train_df.isnull().sum())

# Fill missing values (if any)
train_df.fillna(method='ffill', inplace=True)  
test_df.fillna(method='ffill', inplace=True)
value_df.fillna(method='ffill', inplace=True)

print("\n✅ Missing Values After Cleaning:")
print(train_df.isnull().sum())


# In[16]:


### Validate Data Integrity.


# In[17]:


# Ensure timestamps are sorted
train_df = train_df.sort_values(by="timestamp").reset_index(drop=True)
test_df = test_df.sort_values(by="timestamp").reset_index(drop=True)
value_df = value_df.sort_values(by="timestamp").reset_index(drop=True)


# In[18]:


# Check for duplicate records and remove if necessary
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)
value_df.drop_duplicates(inplace=True)


# In[19]:


#Save Cleaned Data
cleaned_train_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_train.csv"
cleaned_test_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_test.csv"
cleaned_value_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_value.csv"

train_df.to_csv(cleaned_train_path, index=False)
test_df.to_csv(cleaned_test_path, index=False)
value_df.to_csv(cleaned_value_path, index=False)

print("\n✅ Data Preprocessing Completed! Cleaned files saved successfully.")


# In[20]:


# Define file paths for cleaned train.
cleaned_train_path = r"C:\Users\Legion\.jupyter\thesis\cleaned_train.csv"


# In[21]:


# Load cleaned traffic data and AQI site locations
traffic_df = pd.read_csv(cleaned_train_path)
aqi_df = pd.read_csv(aqi_data_path)


# In[22]:


# Convert to GeoDataFrame for spatial operations
traffic_gdf = gpd.GeoDataFrame(traffic_df, geometry=gpd.points_from_xy(traffic_df.longitude, traffic_df.latitude))
aqi_gdf = gpd.GeoDataFrame(aqi_df, geometry=gpd.points_from_xy(aqi_df.longitude, aqi_df.latitude))


# In[23]:


# Set Coordinate Reference System (CRS) to WGS 84 (EPSG:4326)
traffic_gdf.set_crs(epsg=4326, inplace=True)
aqi_gdf.set_crs(epsg=4326, inplace=True)


# In[24]:


#Reproject to UTM Zone 44N (EPSG:32644) for accurate distance calculations
traffic_gdf = traffic_gdf.to_crs(epsg=32644)
site_gdf = site_gdf.to_crs(epsg=32644)


# In[25]:


# Perform a spatial join: Find the nearest AQI site for each traffic data point
joined_df = gpd.sjoin_nearest(traffic_gdf, site_gdf, how="left", distance_col="distance_meters")


# In[26]:


# Convert back to WGS 84 (optional, for mapping)
joined_df = joined_df.to_crs(epsg=4326)


# In[27]:


# Save the spatially joined dataset
joined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv", index=False)
# Display the first few rows
print("\n✅ Spatial Join Completed! First few rows of joined dataset:")
print(joined_df.head())


# In[28]:


##### EDA.


# In[29]:


# Load the spatially joined dataset
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Create a base map centered on Telangana
m = folium.Map(location=[17.5, 78.5], zoom_start=10, tiles="OpenStreetMap")

# Convert latitude & longitude into a format compatible with HeatMap
heat_data = list(zip(joined_df["latitude"], joined_df["longitude"]))

# Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, min_opacity=0.5).add_to(m)

# Save and display the map
heatmap_path = r"C:\Users\Legion\.jupyter\thesis\traffic_heatmap1.html"
m.save(heatmap_path)

print(f"✅ Traffic Heatmap Saved! Open the file in a browser: {heatmap_path}")


# In[30]:


##traffic and aqi station heatmap locating telegana state boundary.
# Load the spatially joined dataset
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Load the India shapefile
india_shapefile = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
gdf = gpd.read_file(india_shapefile)

# Filter to keep only Telangana state
telangana_gdf = gdf[gdf["st_nm"] == "Telangana"]

# Convert CRS to match Folium (WGS 84, EPSG:4326)
telangana_gdf = telangana_gdf.to_crs(epsg=4326)

# Create a base map centered on Telangana
m = folium.Map(location=[17.5, 78.5], zoom_start=7, tiles="OpenStreetMap")

# Add Telangana boundary as a layer
folium.GeoJson(telangana_gdf, name="Telangana Boundary", style_function=lambda x: {
    "color": "black", "weight": 2, "fillOpacity": 0.1
}).add_to(m)

# Convert latitude & longitude into a format compatible with HeatMap
heat_data = list(zip(joined_df["latitude"], joined_df["longitude"]))

# Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, min_opacity=0.5).add_to(m)

# Save and display the map
heatmap_path = r"C:\Users\Legion\.jupyter\thesis\traffic_heatmap2.html"
m.save(heatmap_path)

print(f"✅ Traffic Heatmap Saved! Open the file in a browser: {heatmap_path}")


# In[31]:


## traffic density across telangana.
# Load the spatially joined dataset
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Load the India shapefile
shapefile_path = r"C:\Users\Legion\.jupyter\thesis\India States\Indian_states.shp"
india_map = gpd.read_file(shapefile_path)

# Filter to keep only Telangana state
telangana_map = india_map[india_map["st_nm"] == "Telangana"]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot Telangana boundary
telangana_map.plot(ax=ax, edgecolor="black", color="lightgrey", alpha=0.5)

# Overlay traffic data
ax.scatter(joined_df["longitude"], joined_df["latitude"], c="red", alpha=0.5, label="Traffic Points")

# Set labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traffic Density Across Telangana with GIS Layer")
plt.legend()
plt.show()


# In[32]:


# Load the spatially joined dataset
joined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# Create a scatter plot of traffic locations
plt.figure(figsize=(10, 6))
plt.scatter(joined_df["longitude"], joined_df["latitude"], c="red", alpha=0.5, label="Traffic Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traffic Density Across Telangana")
plt.legend()
plt.show()


# In[33]:


# Group by AQI site name and calculating the average distance to traffic points
aqi_site_avg_distance = joined_df.groupby("Site Name")["distance_meters"].mean().reset_index()

# Plot AQI site pollution trends
plt.figure(figsize=(12, 6))
sns.barplot(x="distance_meters", y="Site Name", data=aqi_site_avg_distance, palette="coolwarm")
plt.xlabel("Average Distance to Traffic (meters)")
plt.ylabel("AQI Site")
plt.title("Average Distance of Traffic to AQI Monitoring Sites")
plt.show()


# In[34]:


# Load AQI dataset
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Print column names to verify structure
print(aqi_df.columns)

# Filter only 'PM2.5' readings from the indicator column
pm25_df = aqi_df[aqi_df["indicator"] == "PM2.5"].copy()

# Rename 'value' column to 'PM2.5' for clarity
pm25_df = pm25_df.rename(columns={"value": "PM2.5"})

# Display the first few rows of PM2.5 data
print(pm25_df.head())


# In[35]:


# Convert Traffic and PM2.5 data into GeoDataFrames
traffic_gdf = gpd.GeoDataFrame(joined_df, geometry=gpd.points_from_xy(joined_df["longitude"], joined_df["latitude"]), crs="EPSG:4326")
pm25_gdf = gpd.GeoDataFrame(pm25_df, geometry=gpd.points_from_xy(pm25_df["longitude"], pm25_df["latitude"]), crs="EPSG:4326")

# Convert to UTM for accurate distance calculations
traffic_gdf = traffic_gdf.to_crs(epsg=32644)
pm25_gdf = pm25_gdf.to_crs(epsg=32644)

# Drop 'index_right' if it exists
if "index_right" in pm25_gdf.columns:
    pm25_gdf = pm25_gdf.drop(columns=["index_right"])

if "index_right" in traffic_gdf.columns:
    traffic_gdf = traffic_gdf.drop(columns=["index_right"])

# Reset index to avoid conflicts
traffic_gdf = traffic_gdf.reset_index(drop=True)
pm25_gdf = pm25_gdf.reset_index(drop=True)

# Find the nearest PM2.5 measurement for each traffic location
joined_pm25_df = gpd.sjoin_nearest(traffic_gdf, pm25_gdf, how="left", distance_col="distance_meters")

# Convert back to WGS 84 for visualization
joined_pm25_df = joined_pm25_df.to_crs(epsg=4326)

# Save the merged dataset
joined_pm25_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_pm25_joined.csv", index=False)

print("\n✅ Merged Traffic & PM2.5 Data Saved! First few rows:")
print(joined_pm25_df.head())


# In[36]:


# Compute correlation between traffic distance & PM2.5 levels
if "distance_meters" in joined_pm25_df.columns and "PM2.5" in joined_pm25_df.columns:
    correlation_value = joined_pm25_df["distance_meters"].corr(joined_pm25_df["PM2.5"])
    print(f"✅ Correlation between Traffic Distance & PM2.5: {correlation_value:.3f}")
else:
    print("⚠️ Error: Required columns not found! Check column names in joined_pm25_df.")


# In[37]:


# Create a base map centered on Telangana
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Add heatmap layer
heat_data = list(zip(joined_df["latitude"], joined_df["longitude"]))
HeatMap(heat_data).add_to(m)

# Save and display the map
m.save(r"C:\Users\Legion\.jupyter\thesis\traffic_heatmap_base map centered on Telangana.html")
print("✅ Traffic Heatmap Saved! Open 'traffic_heatmap.html3' to view.")


# In[38]:


## Visualize the relationship between traffic hotspots and air quality monitoring locations.

# ✅ Load the spatially joined dataset
df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# ✅ Create a base map centered on Telangana
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# ✅ Check if the required columns exist
if "latitude" in df.columns and "longitude" in df.columns:
    # ✅ Add traffic heatmap
    heat_data = list(zip(df["latitude"], df["longitude"]))
    HeatMap(heat_data, radius=15).add_to(m)
    
    # ✅ Add AQI sites as markers
    for _, row in df.iterrows():
        folium.Marker([row["Latitude"], row["Longitude"]],
                      popup=row["Site Name"],
                      icon=folium.Icon(color="red", icon="cloud")).add_to(m)

    # ✅ Save the map
    m.save(r"C:\Users\Legion\.jupyter\traffic_aqi_density_map.html")
    print("✅ Traffic Heatmap with AQI Sites Saved! Open 'traffic_aqi_density_map.html' to view.")
else:
    print("⚠️ Error: Missing 'latitude' or 'longitude' columns in the dataset!")


# In[39]:


## Method 2: Traffic Density Heatmap with AQI Overlays


# In[40]:


# ✅ Load the spatially joined dataset
df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")

# ✅ Create a base map centered on Telangana
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# ✅ Check if the required columns exist
if "latitude" in df.columns and "longitude" in df.columns:
    # ✅ Add traffic heatmap
    heat_data = list(zip(df["latitude"], df["longitude"]))
    HeatMap(heat_data, radius=15).add_to(m)
    
    # ✅ Add AQI sites as markers
    for _, row in df.iterrows():
        folium.Marker([row["Latitude"], row["Longitude"]],
                      popup=row["Site Name"],
                      icon=folium.Icon(color="red", icon="cloud")).add_to(m)

    # ✅ Save the map
    m.save(r"C:\Users\Legion\.jupyter\traffic_aqi_density_map.html")
    print("✅ Traffic Heatmap with AQI Sites Saved! Open 'traffic_aqi_density_map.html' to view.")
else:
    print("⚠️ Error: Missing 'latitude' or 'longitude' columns in the dataset!")


# In[41]:


# Load the dataset
file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_pm25_joined.csv"
df = pd.read_csv(file_path)

# Rename columns for consistency
df.rename(columns={
    "latitude_left": "latitude",
    "longitude_left": "longitude"
}, inplace=True)

# Save the corrected dataset
df.to_csv(file_path, index=False)
print("✅ Column names corrected and dataset saved!")


# In[42]:


# Create a base map centered on Telangana
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Add traffic density heatmap
heat_data = list(zip(df["latitude"], df["longitude"]))
HeatMap(heat_data, radius=15).add_to(m)

# Function to color AQI sites based on PM2.5 levels
def get_color(pm_value):
    if pm_value < 50:
        return "green"
    elif pm_value < 100:
        return "orange"
    else:
        return "red"

# Add AQI site markers with PM2.5 values
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

# Save the map
map_path = r"C:\Users\Legion\.jupyter\thesis\traffic_pm25_heatmap.html"
m.save(map_path)

print(f"✅ Traffic Density Heatmap with PM2.5 Overlays Saved! Open '{map_path}' to view.")


# In[43]:


# Load AQI monitoring site data
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\site.csv")

# Create a base map
aqi_map = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Function to color AQI sites based on pollution level (randomly assigned here)
def get_color(aqi_value):
    if aqi_value < 50:
        return "green"
    elif aqi_value < 100:
        return "orange"
    else:
        return "red"

# Add AQI monitoring sites to the map
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

# Save and display the AQI map
aqi_map.save(r"C:\Users\Legion\.jupyter\thesis\aqi_map_station.html")
print("✅ AQI Map Saved! Open 'aqi_map.html' to view.")


# In[44]:


# Create base map
combined_map = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Add traffic heatmap
HeatMap(heat_data).add_to(combined_map)

# Add AQI monitoring sites with color coding
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

# Save and display combined map
combined_map.save(r"C:\Users\Legion\.jupyter\thesis\combined_traffic_aqi_map.html")
print("✅ Combined Traffic & AQI Map Saved! Open 'combined_traffic_aqi_map2.html' to view.")


# In[45]:


##AQI Heatmap


# In[46]:


# Load AQI data (air quality site locations & AQI values)
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Create a base map centered on Telangana
m = folium.Map(location=[17.5, 78.5], zoom_start=10)

# Add AQI heatmap
heat_data = list(zip(aqi_df["latitude"], aqi_df["longitude"]))
HeatMap(heat_data).add_to(m)

# Save & Display the map
m.save(r"C:\Users\Legion\.jupyter\thesis\aqi_heatmap.html")
print("✅ AQI Heatmap Saved! Open 'aqi_heatmap.html11' to view.")


# In[47]:


##Hourly AQI Trend


# In[48]:


# Load the AQI dataset
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Convert 'date' and 'time' into proper datetime format
aqi_df["datetime"] = pd.to_datetime(aqi_df["date"] + " " + aqi_df["time"], format="%d/%m/%Y %H:%M")

# Ensure 'indicator' column is properly formatted
aqi_df["indicator"] = aqi_df["indicator"].astype(str)

# ✅ Filter data for a specific date (Example: July 11, 2019)
aqi_day = aqi_df[aqi_df["datetime"].dt.date == pd.to_datetime("2019-07-11").date()]

# Plot AQI trend by hour
plt.figure(figsize=(12, 6))
sns.lineplot(data=aqi_day, x="datetime", y="value", hue="indicator", marker="o", palette="tab10")

plt.xlabel("Time of Day")
plt.ylabel("AQI Value")
plt.title("Hourly AQI Trends on July 11, 2019")
plt.xticks(rotation=45)
plt.legend(title="Pollutant Type", loc="upper left", bbox_to_anchor=(1, 1))
plt.show()


# In[49]:


# Plot pollutant distribution for July 11, 2019
plt.figure(figsize=(10, 5))
sns.boxplot(data=aqi_day, x="indicator", y="value")
plt.xlabel("Pollutant Type")
plt.ylabel("AQI Value")
plt.title("Distribution of Different Pollutants on July 11, 2019")
plt.xticks(rotation=45)
plt.show()


# In[50]:


# Average AQI per location
aqi_location_avg = aqi_day.groupby("location")["value"].mean().reset_index()

# Plot AQI by location
plt.figure(figsize=(12, 6))
sns.barplot(data=aqi_location_avg, x="value", y="location", palette="coolwarm")
plt.xlabel("Average AQI")
plt.ylabel("Monitoring Site")
plt.title("Average AQI Levels by Location on July 11, 2019")
plt.show()


# In[51]:


##Merge Traffic & AQI Data on Time


# In[52]:


# Load traffic data (cleaned)
traffic_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\cleaned_train.csv")

# Load AQI data (for July 7, 2019)
aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")

# Convert timestamps to datetime format
traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
aqi_df["datetime"] = pd.to_datetime(aqi_df["date"] + " " + aqi_df["time"], format="%d/%m/%Y %H:%M")

# Extract hour from timestamps for aggregation
traffic_df["hour"] = traffic_df["timestamp"].dt.hour
aqi_df["hour"] = aqi_df["datetime"].dt.hour

# Aggregate traffic data (count of traffic records per hour)
traffic_hourly = traffic_df.groupby("hour").size().reset_index(name="traffic_count")

# Aggregate AQI data (average AQI per hour)
aqi_hourly = aqi_df.groupby(["hour", "indicator"])["value"].mean().reset_index()

# Merge traffic & AQI data on "hour"
combined_df = pd.merge(aqi_hourly, traffic_hourly, on="hour", how="left")

# Save the merged dataset for verification
combined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_merged.csv", index=False)

# Display the first few rows for preview
print("\n✅ Traffic & AQI Merged Data (First 5 rows):")
print(combined_df.head())


# In[53]:


## Visualize Traffic vs. AQI Trends


# In[54]:


# Plot AQI & Traffic trends over time
plt.figure(figsize=(12, 6))

# AQI trend
sns.lineplot(data=combined_df, x="hour", y="value", hue="indicator", marker="o", label="AQI Level")

# Traffic trend
sns.lineplot(data=combined_df, x="hour", y="traffic_count", color="red", marker="s", label="Traffic Count")

plt.xlabel("Hour of Day")
plt.ylabel("AQI Value / Traffic Count")
plt.title("Traffic vs. AQI Trends Throughout the Day")
plt.legend()
plt.show()


# In[55]:


##Correlation Between Traffic & AQI


# In[56]:


# Compute correlation between traffic and AQI values
correlation_matrix = combined_df.pivot(index="hour", columns="indicator", values="value").corrwith(combined_df["traffic_count"])
print("\n📊 Correlation Between Traffic & AQI Indicators:")
print(correlation_matrix)


# In[57]:


# Load datasets
traffic_aqi_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_site_joined.csv")  # Spatially joined dataset
aqi_data = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\Air_Quality_Index_Data.csv")  # AQI readings dataset

# Convert timestamps to datetime format
traffic_aqi_df["timestamp"] = pd.to_datetime(traffic_aqi_df["timestamp"])
aqi_data["datetime"] = pd.to_datetime(aqi_data["date"] + " " + aqi_data["time"], format="%d/%m/%Y %H:%M")

# Extract hour for merging
traffic_aqi_df["hour"] = traffic_aqi_df["timestamp"].dt.hour
aqi_data["hour"] = aqi_data["datetime"].dt.hour

# Merge AQI data into traffic dataset based on hour and nearest location
merged_df = pd.merge(traffic_aqi_df, aqi_data, on="hour", how="left")

# Save merged dataset
merged_file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final.csv"
merged_df.to_csv(merged_file_path, index=False)

print("✅ AQI Data Successfully Merged into Traffic Dataset! Ready for analysis.")


# In[58]:


# Load the merged dataset
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final.csv")

# Fix timestamp format: Strip extra microseconds and convert to datetime
combined_df["timestamp"] = combined_df["timestamp"].str.split(".").str[0]  # Remove microseconds if present
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], errors="coerce")  # Convert properly

# Extract hour for time-based analysis
combined_df["hour"] = combined_df["timestamp"].dt.hour

# Save the corrected dataset
combined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv", index=False)

print("✅ Timestamp Issue Fixed! Dataset saved as 'traffic_aqi_final_fixed.csv'.")


# In[59]:


# Load the dataset
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final.csv")

# Ensure 'timestamp' column is treated as string
combined_df["timestamp"] = combined_df["timestamp"].astype(str)

# Remove any extra spaces or unexpected characters
combined_df["timestamp"] = combined_df["timestamp"].str.strip()

# Try different formats and clean microseconds if necessary
def clean_timestamp(ts):
    try:
        # Remove microseconds if present
        if "." in ts:
            ts = ts.split(".")[0]
        # Convert to datetime
        return pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    except:
        return pd.NaT  # Assign NaT (Not a Time) if parsing fails

# Apply timestamp cleaning function
combined_df["timestamp"] = combined_df["timestamp"].apply(clean_timestamp)

# Drop any rows with NaT timestamps (optional, if needed)
combined_df = combined_df.dropna(subset=["timestamp"])

# Extract hour from timestamp for time-based analysis
combined_df["hour"] = combined_df["timestamp"].dt.hour

# Save the cleaned dataset
fixed_file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv"
combined_df.to_csv(fixed_file_path, index=False)

print("✅ Timestamp Issue Fixed! Cleaned dataset saved as 'traffic_aqi_final_fixed.csv'.")


# In[60]:


# Load the dataset
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv")

# Convert timestamp to datetime format
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

# Count traffic events per hour
traffic_count_per_hour = combined_df.groupby("hour").size().reset_index(name="traffic_count")

# Merge back into the dataset
combined_df = pd.merge(combined_df, traffic_count_per_hour, on="hour", how="left")

# Save the updated dataset
combined_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv", index=False)

print("✅ Traffic count successfully added to the dataset!")


# In[61]:


# Reload the fixed dataset
combined_df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv")

# Convert timestamp to datetime format
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

# Aggregate AQI values per hour (average to remove duplicates)
aggregated_df = combined_df.groupby(["hour", "indicator"], as_index=False).agg({"value": "mean", "traffic_count": "sum"})

# Pivot after aggregation
pivot_df = aggregated_df.pivot(index="hour", columns="indicator", values="value")

# Compute correlation between traffic count and AQI values
correlation_matrix = pivot_df.corrwith(aggregated_df.groupby("hour")["traffic_count"].sum())

# Save correlation results to CSV (for inspection)
correlation_matrix.to_csv(r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_correlation.csv")

# Print correlation results
print("\n✅ Traffic & AQI Correlation Results:")
print(correlation_matrix)

# Plot AQI & Traffic trends over time
plt.figure(figsize=(12, 6))

# AQI trend
sns.lineplot(data=combined_df, x="hour", y="value", hue="indicator", marker="o", label="AQI Level")

# Traffic trend
sns.lineplot(data=combined_df, x="hour", y="traffic_count", color="red", marker="s", label="Traffic Count")

plt.xlabel("Hour of Day")
plt.ylabel("AQI Value / Traffic Count")
plt.title("Traffic vs. AQI Trends Throughout the Day")
plt.legend()
plt.show()


# In[62]:


##Correlation Analysis , Fix: Normalize Traffic & AQI Data


# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale 'traffic_count' to range 0-1
aggregated_df["traffic_count_scaled"] = scaler.fit_transform(aggregated_df[["traffic_count"]])

# Plot AQI & Traffic trends over time
plt.figure(figsize=(14, 6))

# AQI trend
sns.lineplot(data=aggregated_df, x="hour", y="value", hue="indicator", marker="o")

# Traffic trend (scaled)
sns.lineplot(data=aggregated_df, x="hour", y="traffic_count_scaled", color="red", marker="s", label="Traffic Count (Scaled)")

plt.xlabel("Hour of Day")
plt.ylabel("AQI Value / Scaled Traffic Count")
plt.title("Traffic vs. AQI Trends Throughout the Day")

# Move legend outside the plot
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Adjust y-axis limits for better visibility
plt.ylim(0, max(aggregated_df["value"].max(), 1))  # Since traffic is scaled, upper limit is 1

plt.show()


# In[64]:


##ML modeling.


# In[65]:


# Load dataset
file_path = r"C:\Users\Legion\.jupyter\traffic_aqi_final_fixed.csv"
df = pd.read_csv(file_path)

# Convert timestamp to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Print dataset info
print("🔍 Dataset Info:")
print(df.info())

# Display first few rows
print("\n📊 Sample Data:")
print(df.head())


# In[66]:


# Check for missing values
print("\n❗ Missing Values Before Handling:")
print(df.isnull().sum())

# Fill missing values (use forward fill for time-series data)
df.fillna(method='ffill', inplace=True)

# Verify missing values are handled
print("\n✅ Missing Values After Handling:")
print(df.isnull().sum())


# In[67]:


# Load the dataset
file_path = r"C:\Users\Legion\.jupyter\thesis\traffic_aqi_final_fixed.csv"
df = pd.read_csv(file_path)

# Convert timestamp to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract hour for time-based modeling
df["hour"] = df["timestamp"].dt.hour

# Pivot AQI data (convert 'indicator' column into separate pollutant columns)
df_pivot = df.pivot_table(index=["timestamp", "hour", "traffic_count"], 
                          columns="indicator", values="value").reset_index()

# Save the pivoted dataset
df_pivot.to_csv(r"C:\Users\Legion\.jupyter\thesis\pivoted_data.csv", index=False)

print("✅ AQI Data Successfully Pivoted! Saved as 'pivoted_data.csv'.")


# In[68]:


# Reload the pivoted dataset
df_ml = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\pivoted_data.csv")

# Select relevant features
selected_features = ["hour", "traffic_count", "NO2", "PM2.5", "Ozone", "CO", "SO2"]

# Filter dataset
df_ml = df_ml[selected_features]

# Save processed dataset
df_ml.to_csv(r"C:\Users\Legion\.jupyter\thesis\processed_data.csv", index=False)

print("✅ Feature Selection Completed! Processed data saved as 'processed_data.csv'.")


# In[69]:


# Load dataset
df_ml = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\processed_data.csv")

# Initialize scaler
scaler = MinMaxScaler()

# Scale data
df_scaled = pd.DataFrame(scaler.fit_transform(df_ml), columns=df_ml.columns)

# Save scaled dataset
df_scaled.to_csv(r"C:\Users\Legion\.jupyter\thesis\scaled_data.csv", index=False)

print("✅ Data Scaling Completed! Saved as 'scaled_data.csv'.")


# In[70]:


# train and evaluate random forest model.


# In[71]:


# Load the processed dataset
df = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\scaled_data.csv")

# Define input features (X) and target variable (y)
X = df.drop(columns=["PM2.5"])  # Predicting PM2.5 levels
y = df["PM2.5"]

# Split data into train & test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Random Forest Model Evaluation:")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Feature Importance
importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)

print("\n🔍 Feature Importance Ranking:")
print(importances)

# Save model predictions
predictions_df = pd.DataFrame({"Actual_PM2.5": y_test, "Predicted_PM2.5": y_pred})
predictions_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\rf_predictions.csv", index=False)

print("\n✅ Predictions saved as 'rf_predictions.csv'")


# In[72]:


# train and evaluate LSTM


# In[73]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Define input shape (assuming X_train is already shaped properly)
input_shape = (X_train.shape[1], 1)  # Adjust based on data preprocessing

# ✅ Corrected Model
model = Sequential([
    Input(shape=input_shape),  # Define input layer explicitly
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Print model summary
model.summary()


# In[74]:


import numpy as np

# Ensure X_train is in the correct shape: (samples, time_steps, features)
X_train = np.expand_dims(X_train, axis=2)  # Adds a 3rd dimension
X_test = np.expand_dims(X_test, axis=2)

# Print shape to verify
print("✅ Reshaped X_train:", X_train.shape)  # Should be (samples, time_steps, features)
print("✅ Reshaped X_test:", X_test.shape)


# In[75]:


# Correct X_train and X_test shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], -1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))

# Print new shapes
print("✅ Corrected X_train Shape:", X_train.shape)  # Should be (samples, time_steps, features)
print("✅ Corrected X_test Shape:", X_test.shape)


# In[76]:


import numpy as np

# Ensure the correct shape: (samples, time_steps, features)
time_steps = 6  # Adjust this based on your dataset's sequence requirement
features = 1

X_train = X_train.reshape(X_train.shape[0], time_steps, features)
X_test = X_test.reshape(X_test.shape[0], time_steps, features)

# Print the new shape
print("✅ Corrected X_train Shape:", X_train.shape)  # Should be (samples, time_steps, features)
print("✅ Corrected X_test Shape:", X_test.shape)


# In[77]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Correct input shape
input_shape = (time_steps, features)  # (6, 1)

# ✅ Define LSTM Model
model = Sequential([
    Input(shape=input_shape),  # Corrected input layer
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Print model summary
model.summary()


# In[78]:


# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)





# In[88]:


from tensorflow.keras.models import load_model

# ✅ Load the trained LSTM model with a custom loss function
lstm_model = load_model(
    r"C:\Users\Legion\.jupyter\thesis\lstm_final_model.h5",
    compile=False  # Prevents the error
)

# ✅ Recompile the model explicitly
lstm_model.compile(optimizer="adam", loss="mean_squared_error")

print("✅ LSTM Model Loaded and Recompiled Successfully!")


# In[90]:


import pandas as pd
import numpy as np

# ✅ Make Predictions
y_pred_lstm = lstm_model.predict(X_test)

# ✅ Save LSTM Predictions
lstm_predictions_df = pd.DataFrame({"Actual_PM2.5": y_test, "Predicted_PM2.5": y_pred_lstm.flatten()})
lstm_predictions_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\lstm_predictions.csv", index=False)

print("✅ LSTM Predictions Saved Successfully at 'lstm_predictions.csv'!")


# In[79]:


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LSTM Training Progress (Loss Curve)")
plt.legend()
plt.grid()
plt.show()


# In[80]:


#Compare Performance (RF vs. LSTM).


# In[91]:


# ✅ Load Predictions
rf_preds = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\rf_predictions.csv")  # ✅ Correct
lstm_preds = pd.read_csv(r"C:\Users\Legion\.jupyter\thesis\lstm_predictions.csv")  # ✅ Correct

# ✅ Compute Performance Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {"Random Forest": rf_preds, "LSTM": lstm_preds}
results = {}

for model_name, df in models.items():
    mae = mean_absolute_error(df["Actual_PM2.5"], df["Predicted_PM2.5"])
    rmse = np.sqrt(mean_squared_error(df["Actual_PM2.5"], df["Predicted_PM2.5"]))
    r2 = r2_score(df["Actual_PM2.5"], df["Predicted_PM2.5"])
    
    results[model_name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

# ✅ Convert results to DataFrame
results_df = pd.DataFrame(results).T

# ✅ Save results
results_df.to_csv(r"C:\Users\Legion\.jupyter\thesis\model_comparison.csv")

# ✅ Print comparison table
print("\n✅ Model Performance Comparison:")
print(results_df)


# In[92]:


#Visualize Predictions


# In[93]:


# Plot Actual vs Predicted AQI (PM2.5)
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


# In[94]:


##Hyperparameter Tuning.


# In[95]:


# RANDOM FOREST


# In[96]:


# Ensure X_train and X_test are 2D (Remove extra dimensions)
X_train = X_train.reshape(X_train.shape[0], -1)  # Convert to (samples, features)
X_test = X_test.reshape(X_test.shape[0], -1)

# Define parameter grid for tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None]
}

# Initialize RF model
rf = RandomForestRegressor(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters
print("\n✅ Best Parameters for Random Forest:", grid_search.best_params_)


# In[97]:


# Create the Random Forest model
rf_model = RandomForestRegressor()

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Max depth of trees
    'min_samples_split': [2, 5, 10],  # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],    # Min samples at leaf node
    'max_features': ['sqrt', 'log2', None]  # Correct values for max_features
}


# Perform Grid Search with cross-validation (disable parallelization by setting n_jobs=1)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1)

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Print the best hyperparameters
print("\n✅ Best Hyperparameters for Random Forest:")
print(grid_search_rf.best_params_)

# Save the best parameters
best_rf_params = grid_search_rf.best_params_
with open(r"C:\Users\Legion\.jupyter\thesis\rf_best_params.txt", "w") as f:
    f.write(str(best_rf_params))


# In[98]:


# Create and train the Random Forest model with the best hyperparameters
rf_model = RandomForestRegressor(
    max_depth=10, 
    max_features=None, 
    min_samples_leaf=4, 
    min_samples_split=10, 
    n_estimators=100, 
    random_state=42
)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print evaluation metrics
print("Random Forest Model - MSE:", mse_rf)
print("Random Forest Model - R2:", r2_rf)


# In[99]:


# Plot predictions vs actual values for Random Forest
plt.figure(figsize=(10,6))
plt.plot(y_test, label="Actual Values", color='blue', linestyle='--')
plt.plot(y_pred_rf, label="Predicted Values (Random Forest)", color='red')
plt.legend()
plt.xlabel("Data Points")
plt.ylabel("AQI Values")
plt.title("Predicted vs Actual AQI Values (Random Forest)")
plt.show()


# In[104]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Scatter plot for actual vs predicted values
plt.scatter(y_test, y_pred_rf, color="red", alpha=0.5, label="Predicted vs Actual")

# Plot a perfect prediction reference line (y = x)
min_val = min(min(y_test), min(y_pred_rf))
max_val = max(max(y_test), max(y_pred_rf))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="blue", label="Perfect Prediction (y=x)")

plt.xlabel("Actual AQI Values")
plt.ylabel("Predicted AQI Values")
plt.title("Predicted vs. Actual AQI Values (Random Forest)")
plt.legend()
plt.grid(True)
plt.show()


# In[100]:


#LSTM Tuning


# In[101]:


import numpy as np

# ✅ Ensure input shape matches LSTM format: (samples, time_steps, features)
time_steps = 1  # Since there's no time-series sequence
features = X_train.shape[1]  # Number of input features

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], time_steps, features))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], time_steps, features))

# Print new shape for confirmation
print("✅ X_train_reshaped:", X_train_reshaped.shape)  # Expected: (samples, 1, features)
print("✅ X_test_reshaped:", X_test_reshaped.shape)


# In[102]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Clear previous session
from tensorflow.keras import backend as K
K.clear_session()

# ✅ Define the LSTM model correctly
model = Sequential([
    Input(shape=(time_steps, features)),  # Use Input layer instead of input_shape in LSTM
    LSTM(100, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(100, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# ✅ Step 3: Train the Model with Reshaped Input
history = model.fit(
    X_train_reshaped, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)


# In[103]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

# Custom wrapper class for Keras model
class KerasModelWrapper(BaseEstimator):
    def __init__(self, units=50, dropout_rate=0.2, learning_rate=0.001):
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None  # Define model placeholder

    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),  # Use Input layer instead of input_shape in LSTM
            LSTM(self.units, activation='relu', return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def fit(self, X, y, batch_size=32, epochs=50, **kwargs):
        # ✅ Ensure correct LSTM input shape
        time_steps = 1  # Adjust based on dataset
        features = X.shape[1]
        X_reshaped = np.reshape(X, (X.shape[0], time_steps, features))

        self.model = self.build_model(input_shape=(time_steps, features))
        self.model.fit(X_reshaped, y, batch_size=batch_size, epochs=epochs, verbose=0, **kwargs)
        return self

    def predict(self, X):
        X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        return self.model.predict(X_reshaped)



# In[105]:


# Define parameter grid for tuning
param_grid = {
    "units": [50, 100, 150],
    "dropout_rate": [0.2, 0.3, 0.4],
    "learning_rate": [0.001, 0.0005, 0.0001]
}

# Create wrapper model
wrapper_model = KerasModelWrapper()

# Perform Grid Search with cross-validation (disable parallelization by setting n_jobs=1)
grid_search = GridSearchCV(estimator=wrapper_model, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=1)

# ✅ Run Grid Search
grid_search.fit(X_train, y_train)  # No need for reshaping here, it's handled in the wrapper

# Print best parameters
print("\n✅ Best Hyperparameters for LSTM:")
print(grid_search.best_params_)

# Save best parameters to a text file
best_params = grid_search.best_params_
with open(r"C:\Users\Legion\.jupyter\thesis\lstm_best_params.txt", "w") as f:
    f.write(str(best_params))


# In[106]:


# ✅ Define the final LSTM model with best parameters
final_model = Sequential([
    Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    LSTM(150, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(150, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer
])

# ✅ Compile the final model with optimized learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
final_model.compile(optimizer=optimizer, loss='mse')

# ✅ Train the final LSTM model
history = final_model.fit(
    X_train_reshaped, y_train,
    validation_data=(X_test_reshaped, y_test),
    epochs=100,  # Train for more epochs to fully leverage best params
    batch_size=32,
    verbose=1
)

# ✅ Save the final trained model
final_model.save(r"C:\Users\Legion\.jupyter\thesis\lstm_final_model.h5")

print("\n✅ Final LSTM Model Training Completed & Saved as 'lstm_final_model.h5'!")


# In[107]:


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions
y_pred = final_model.predict(X_test_reshaped)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Final Model Performance:")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")


# In[108]:


# After training your LSTM model
y_pred_lstm = model.predict(X_test_reshaped)  # Use the reshaped test set for predictions

# Plot predictions vs actual values for LSTM
plt.figure(figsize=(10,6))
plt.plot(y_test, label="Actual Values", color='blue', linestyle='--')  # Actual values (LSTM)
plt.plot(y_pred_lstm, label="Predicted Values (LSTM)", color='red')  # Predicted values (LSTM)
plt.legend()
plt.title("Predicted vs Actual AQI Values (LSTM)")
plt.xlabel("Data Points")
plt.ylabel("AQI Values")
plt.show()


# In[110]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

# Scatter plot for actual vs predicted values
plt.scatter(y_test, y_pred_lstm, color="red", alpha=0.5, label="Predicted vs Actual")

# Perfect prediction reference line (y = x)
min_val = min(min(y_test), min(y_pred_lstm))
max_val = max(max(y_test), max(y_pred_lstm))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="blue", label="Perfect Prediction (y=x)")

plt.xlabel("Actual AQI Values")
plt.ylabel("Predicted AQI Values")
plt.title("Predicted vs. Actual AQI Values (LSTM)")
plt.legend()
plt.grid(True)
plt.show()


# In[114]:


pip install --upgrade scikit-learn


# In[116]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Ensure y_test and y_pred_lstm are 1D arrays
y_test = np.array(y_test).flatten()
y_pred_lstm = np.array(y_pred_lstm).flatten()

# Compute Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred_lstm)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lstm))  # Manually computing RMSE
r2 = r2_score(y_test, y_pred_lstm)

print(f"📉 MAE: {mae:.2f}")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")


# In[120]:


tolerance_margin = 0.05

# Flatten the arrays to ensure they are 1D
y_test = np.array(y_test).flatten()
y_pred_rf = np.array(y_pred_rf).flatten()
y_pred_lstm = np.array(y_pred_lstm).flatten()


# Get the absolute difference between the predicted and actual values
absolute_error_rf = np.abs(y_pred_rf - y_test)
absolute_error_lstm = np.abs(y_pred_lstm - y_test)

# Calculate the percentage of predictions within the tolerance margin
accuracy_rf = np.mean(absolute_error_rf <= (tolerance_margin * y_test)) * 100
accuracy_lstm = np.mean(absolute_error_lstm <= (tolerance_margin * y_test)) * 100

print(f"✅ Random Forest Model Accuracy: {accuracy_rf:.2f}%")
print(f"✅ LSTM Model Accuracy: {accuracy_lstm:.2f}%")


# In[121]:


# Calculate MSE, RMSE, R2, and MAE for both models
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

mse_lstm = mean_squared_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

# Print the results
print(f"Random Forest - MSE: {mse_rf:.4f}, RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}, MAE: {mae_rf:.4f}")
print(f"LSTM - MSE: {mse_lstm:.4f}, RMSE: {rmse_lstm:.4f}, R²: {r2_lstm:.4f}, MAE: {mae_lstm:.4f}")


# In[111]:


# Plot residuals for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, y_test - y_pred_rf, color='blue', label='Random Forest')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted AQI')
plt.ylabel('Residuals')
plt.title('Residual Plot (Random Forest)')
plt.legend()
plt.show()

# Flatten the arrays to ensure they are 1D
y_test_flat = y_test.flatten()
y_pred_lstm_flat = y_pred_lstm.flatten()

# Plot residuals for LSTM
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lstm_flat, y_test_flat - y_pred_lstm_flat, color='green', label='LSTM')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted AQI')
plt.ylabel('Residuals')
plt.title('Residual Plot (LSTM)')
plt.legend()
plt.show()



# In[124]:


# Ensure LSTM predictions are in 1D format
y_pred_lstm_flat = y_pred_lstm.flatten()  
y_test_flat = y_test.flatten()  

# Plot Prediction vs Error for LSTM
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lstm_flat, y_test_flat - y_pred_lstm_flat, color='green', label='LSTM')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted AQI')
plt.ylabel('Prediction Error (Actual - Predicted)')
plt.legend()
plt.show()


# In[125]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_lstm = mean_absolute_error(y_test_flat, y_pred_lstm_flat)
rmse_lstm = np.sqrt(mean_squared_error(y_test_flat, y_pred_lstm_flat))
r2_lstm = r2_score(y_test_flat, y_pred_lstm_flat)

print(f"📉 LSTM MAE: {mae_lstm:.4f}")
print(f"📉 LSTM RMSE: {rmse_lstm:.4f}")
print(f"📈 LSTM R² Score: {r2_lstm:.4f}")


# In[126]:


lstm_results_df = pd.DataFrame({
    "Actual_PM2.5": y_test_flat,
    "Predicted_PM2.5": y_pred_lstm_flat
})

lstm_results_df.to_csv("lstm_predictions.csv", index=False)
print("✅ LSTM Predictions Saved!")


# In[122]:


# Plot Prediction vs Error for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, y_test - y_pred_rf, color='blue', label='Random Forest')
plt.xlabel('Predicted AQI')
plt.ylabel('Prediction Error (AQI - Predicted)')
plt.title('Prediction vs Error (Random Forest)')
plt.legend()
plt.show()

# Plot Prediction vs Error for LSTM
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lstm_flat, y_test_flat - y_pred_lstm_flat, color='green', label='LSTM')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted AQI')
plt.ylabel('Residuals')
plt.title('Residual Plot (LSTM)')
plt.legend()
plt.show()



# In[113]:


# Plot MSE for both models
mse_values = [mse_rf, mse_lstm]
labels = ['Random Forest', 'LSTM']

plt.figure(figsize=(10, 6))
plt.bar(labels, mse_values, color=['blue', 'green'])
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Model Performance Comparison (MSE)')
plt.show()

# Plot R² for both models
r2_values = [r2_rf, r2_lstm]

plt.figure(figsize=(10, 6))
plt.bar(labels, r2_values, color=['blue', 'green'])
plt.ylabel('R²')
plt.title('Model Performance Comparison (R²)')
plt.show()


# In[114]:


# Example of plotting learning curves
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=0)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title('Learning Curves (LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[115]:


# Calculate cumulative error for Random Forest
cumulative_error_rf = np.cumsum(np.abs(y_test - y_pred_rf))

# Plot cumulative error
plt.figure(figsize=(10, 6))
plt.plot(cumulative_error_rf, label='Cumulative Error (Random Forest)', color='blue')
plt.xlabel('Data Points')
plt.ylabel('Cumulative Error')
plt.title('Cumulative Error Plot (Random Forest)')
plt.legend()
plt.show()

# Calculate cumulative error for LSTM
cumulative_error_lstm = np.cumsum(np.abs(y_test - y_pred_lstm))

# Plot cumulative error for LSTM
plt.figure(figsize=(10, 6))
plt.plot(cumulative_error_lstm, label='Cumulative Error (LSTM)', color='green')
plt.xlabel('Data Points')
plt.ylabel('Cumulative Error')
plt.title('Cumulative Error Plot (LSTM)')
plt.legend()
plt.show()


# In[120]:


import os

# List all .h5 files in the current directory
model_files = [f for f in os.listdir() if f.endswith(".h5")]
print("Available Model Files:", model_files)


# In[118]:


try:
    lstm_model.save("lstm_final_model.h5")
    print("✅ LSTM Model Saved Successfully!")
except NameError:
    print("⚠️ Error: The LSTM model is not defined. Ensure it’s trained before saving.")


# In[ ]:


import joblib

# Save Random Forest model
joblib.dump(rf_model, "rf_model.pkl")

# ✅ Check and save LSTM model
try:
    final_model.save("lstm_model.h5")  # Use the correct model name
    print("✅ Models saved successfully!")
except NameError:
    print("⚠️ Error: The LSTM model is not defined. Ensure it's trained before saving.")


# In[127]:


pip install flask joblib tensorflow pandas numpy matplotlib


# In[142]:


lstm_model.save("lstm_model.keras")  # Save in recommended format


# In[143]:


from tensorflow.keras.models import load_model

# Load the model
lstm_model = load_model("lstm_model.keras", compile=False)

# Recompile the model to avoid optimizer issues
lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

print("✅ LSTM Model Loaded and Compiled Successfully!")


# In[128]:


import joblib
from tensorflow.keras.models import load_model

# Save Random Forest model
joblib.dump(rf_model, "rf_model.pkl")

# Save LSTM model
lstm_model.save("lstm_model.h5")

print("✅ Models Saved Successfully!")


# In[138]:


lstm_model.save("lstm_model.keras")  # ✅ Convert to .keras format
print("✅ Model saved as 'lstm_model.keras'")


# In[139]:


lstm_model = load_model("lstm_model.keras")


# In[140]:


from tensorflow.keras.models import load_model

# Load the model
lstm_model = load_model("lstm_model.keras")

# Recompile the model
lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# In[149]:


import os
import shutil

# Define the full path for the models directory
base_dir = r"C:\Users\Legion\.jupyter\thesis\AQI_Prediction_WebApp"
model_dir = os.path.join(base_dir, "models")

# ✅ Ensure the base directory exists before creating the models folder
os.makedirs(model_dir, exist_ok=True)

# List of model files to move
model_files = ["lstm_model.keras", "lstm_model.h5", "best_lstm_model.h5", "rf_model.pkl"]

# Move each file to the new location
for model in model_files:
    src_path = os.path.join(r"C:\Users\Legion", model)  # Assuming models are in C:\Users\Legion
    dest_path = os.path.join(model_dir, model)

    if os.path.exists(src_path):  # Check if the file exists
        shutil.move(src_path, dest_path)
        print(f"✅ Moved {model} to {dest_path}")
    else:
        print(f"⚠️ {model} not found, skipping.")

print("\n✅ All available model files moved successfully!")


# In[ ]:




