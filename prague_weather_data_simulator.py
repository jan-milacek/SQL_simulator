import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math
import csv

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the parameters
start_date = datetime(2024, 10, 1, 0, 0)  # October 1, 2024
end_date = datetime(2025, 3, 31, 23, 59)  # March 31, 2025
interval_minutes = 5
device_ids = ["weather_station_001", "weather_station_002", "weather_station_003"]
sensor_types = ["temperature", "humidity", "pressure", "wind_speed", "rainfall", "light_level"]

# Prague seasonal parameters (approximate)
# Monthly average temperatures in Prague (°C)
monthly_avg_temp = {
    10: 9,    # October
    11: 4,    # November
    12: 0,    # December
    1: -1,    # January
    2: 0,     # February
    3: 4      # March
}

# Monthly average humidity in Prague (%)
monthly_avg_humidity = {
    10: 75,   # October
    11: 80,   # November
    12: 85,   # December
    1: 85,    # January
    2: 80,    # February
    3: 75     # March
}

# Function to generate data with daily and seasonal patterns
def generate_sensor_value(timestamp, sensor_type, device_id):
    # Get hour of day (0-23) for daily patterns
    hour = timestamp.hour
    
    # Get day of year (0-365) for seasonal patterns
    day_of_year = timestamp.timetuple().tm_yday
    month = timestamp.month
    
    # Base value with random noise
    noise = np.random.normal(0, 1)
    
    # Add some device-specific bias
    device_bias = 0
    if device_id == "weather_station_001":
        device_bias = 0.5
    elif device_id == "weather_station_002":
        device_bias = -0.2
    elif device_id == "weather_station_003":
        device_bias = 0.1
    
    if sensor_type == "temperature":
        # Seasonal component for temperature
        monthly_base = monthly_avg_temp[month]
        
        # Daily cycle: coolest at ~5AM, warmest at ~2PM
        daily_cycle = -5 * math.cos((hour - 14) * 2 * math.pi / 24)
        
        # Random variations
        random_component = noise * 2.5
        
        return round(monthly_base + daily_cycle + device_bias + random_component, 1)
    
    elif sensor_type == "humidity":
        # Seasonal component for humidity
        monthly_base = monthly_avg_humidity[month]
        
        # Daily cycle: highest in early morning, lowest in afternoon
        daily_cycle = 10 * math.cos((hour - 14) * 2 * math.pi / 24)
        
        # Random variations
        random_component = noise * 5
        
        # Ensure humidity stays in valid range
        humidity = monthly_base + daily_cycle + device_bias * 5 + random_component
        return round(min(max(humidity, 20), 100), 1)
    
    elif sensor_type == "pressure":
        # Base pressure around 1013 hPa with seasonal variations
        base_pressure = 1013
        seasonal_var = 5 * math.sin(day_of_year * 2 * math.pi / 365)
        
        # Random daily variations
        daily_var = 10 * np.random.normal(0, 0.5)
        
        return round(base_pressure + seasonal_var + daily_var + device_bias * 2, 1)
    
    elif sensor_type == "wind_speed":
        # Wind tends to be higher in winter months
        seasonal_factor = 1.0
        if month in [11, 12, 1, 2]:  # Winter months
            seasonal_factor = 1.5
        
        # Wind also tends to pick up during the day
        time_factor = 0.7 + 0.6 * math.sin((hour - 12) * math.pi / 12)
        
        base_speed = 5 * seasonal_factor * time_factor
        random_component = abs(noise) * 3  # Using abs to ensure non-negative
        
        return round(base_speed + random_component + device_bias, 1)
    
    elif sensor_type == "rainfall":
        # More rainfall in autumn and early spring
        seasonal_prob = 0.1  # Base probability
        if month in [10, 11, 3]:  # Autumn and early spring
            seasonal_prob = 0.3
        elif month in [12, 1, 2]:  # Winter (some as snow, less volume)
            seasonal_prob = 0.2
        
        # Generate rainfall events
        if random.random() < seasonal_prob:
            # If it's raining, generate amount (higher in autumn/spring)
            base_amount = 0.5
            if month in [10, 11, 3]:
                base_amount = 1.0
            
            amount = base_amount * abs(noise * 2)
            return round(amount, 1)
        else:
            # No rain
            return 0.0
    
    elif sensor_type == "light_level":
        # Light level depends on time of day (daylight)
        if 7 <= hour <= 17:  # Daytime
            # Shorter daylight in winter
            if month in [11, 12, 1, 2]:
                if hour < 8 or hour > 16:
                    return round(random.uniform(0, 100), 1)
                else:
                    return round(random.uniform(500, 1000) + device_bias * 20, 1)
            else:
                return round(random.uniform(800, 1200) + device_bias * 20, 1)
        else:  # Night
            return round(random.uniform(0, 10), 1)

# Generate timestamps for the entire period
timestamps = []
current_time = start_date
while current_time <= end_date:
    timestamps.append(current_time)
    current_time += timedelta(minutes=interval_minutes)

# Generate the data
data = []
for ts in timestamps:
    for device_id in device_ids:
        for sensor_type in sensor_types:
            value = generate_sensor_value(ts, sensor_type, device_id)
            data.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": device_id,
                "sensor_type": sensor_type,
                "value": value
            })

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_filename = "prague_weather_iot_data_6months.csv"
df.to_csv(csv_filename, index=False)

# Print summary info
total_records = len(df)
time_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
total_days = (end_date - start_date).days
device_count = len(device_ids)
sensor_count = len(sensor_types)
file_size_estimate = total_records * 100 / (1024 * 1024)  # Rough estimate in MB

print(f"Generated {csv_filename}")
print(f"Time period: {time_range} ({total_days} days)")
print(f"Records: {total_records:,}")
print(f"Devices: {device_count}")
print(f"Sensor types: {sensor_count}")
print(f"Measurement frequency: Every {interval_minutes} minutes")
print(f"Estimated file size: ~{file_size_estimate:.1f} MB")

# Show a few sample rows
print("\nSample data:")
print(df.head(10))

# Print some statistics by sensor type for validation
print("\nSensor statistics (mean values by month):")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month'] = df['timestamp'].dt.month
monthly_stats = df.groupby(['month', 'sensor_type'])['value'].mean().unstack()
print(monthly_stats.round(2))