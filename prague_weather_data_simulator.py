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

# Define anomaly types
class AnomalyGenerator:
    @staticmethod
    def no_anomaly(value):
        return value
    
    @staticmethod
    def sudden_spike(value):
        # Sudden significant increase
        multiplier = random.uniform(3, 8)
        return value * multiplier if value >= 0 else value / multiplier
    
    @staticmethod
    def sudden_drop(value):
        # Sudden significant decrease
        multiplier = random.uniform(0.1, 0.4)
        return value * multiplier if value >= 0 else value / multiplier
    
    @staticmethod
    def stuck_value(value):
        # Return a fixed value (simulating a stuck sensor)
        # Will be applied to multiple consecutive readings
        return value  # Initial value is returned, then kept constant
    
    @staticmethod
    def high_noise(value):
        # Add excessive noise
        noise_factor = random.uniform(1.5, 3.0)
        # Use abs() to ensure scale is positive
        scale = abs(value * noise_factor * 0.15)
        noise = np.random.normal(0, scale)
        return value + noise
    
    @staticmethod
    def gradual_drift(value, drift_factor):
        # Gradual drift (will increase/decrease over time)
        return value * (1 + drift_factor)
    
    @staticmethod
    def oscillation(value, oscillation_state):
        # Oscillating between high and low values
        amplitude = abs(value * 0.5)  # Ensure amplitude is positive
        if oscillation_state:
            return value + amplitude
        else:
            return value - amplitude

# Schedule anomalies for each device and sensor
def generate_anomaly_schedule():
    anomalies = []
    
    # List of possible anomalies with their durations and probabilities
    anomaly_types = [
        {"type": "sudden_spike", "duration_minutes": 5, "probability": 0.005},  # Rare, very short
        {"type": "sudden_drop", "duration_minutes": 5, "probability": 0.005},   # Rare, very short
        {"type": "stuck_value", "duration_minutes": 180, "probability": 0.001},  # Very rare, longer duration
        {"type": "high_noise", "duration_minutes": 120, "probability": 0.002},   # Rare, medium duration
        {"type": "gradual_drift", "duration_minutes": 1440, "probability": 0.0005},  # Very rare, long duration (1 day)
        {"type": "oscillation", "duration_minutes": 60, "probability": 0.003}    # Rare, medium duration
    ]
    
    # Generate timestamps for the entire period
    current_time = start_date
    while current_time <= end_date:
        # For each device and sensor combination
        for device_id in device_ids:
            for sensor_type in sensor_types:
                # Check each anomaly type
                for anomaly_config in anomaly_types:
                    # Determine if an anomaly starts at this timestamp
                    if random.random() < anomaly_config["probability"]:
                        anomaly = {
                            "device_id": device_id,
                            "sensor_type": sensor_type,
                            "anomaly_type": anomaly_config["type"],
                            "start_time": current_time,
                            "end_time": current_time + timedelta(minutes=anomaly_config["duration_minutes"]),
                            "drift_factor": random.uniform(0.001, 0.01) if anomaly_config["type"] == "gradual_drift" else 0,
                            "oscillation_state": True,  # For oscillation anomaly
                            "stuck_value": None  # For stuck_value anomaly, will be set on first occurrence
                        }
                        anomalies.append(anomaly)
        
        current_time += timedelta(minutes=interval_minutes)
    
    return anomalies

# Generate the anomaly schedule once
anomaly_schedule = generate_anomaly_schedule()
print(f"Generated {len(anomaly_schedule)} anomalies")

# Function to check if a particular reading should have an anomaly
def get_anomaly(timestamp, device_id, sensor_type):
    for anomaly in anomaly_schedule:
        if (anomaly["device_id"] == device_id and 
            anomaly["sensor_type"] == sensor_type and 
            anomaly["start_time"] <= timestamp <= anomaly["end_time"]):
            return anomaly
    return None

# Function to generate data with daily and seasonal patterns and anomalies
def generate_sensor_value(timestamp, sensor_type, device_id, prev_value=None, prev_anomaly=None):
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
    
    # Generate the normal value first
    if sensor_type == "temperature":
        # Seasonal component for temperature
        monthly_base = monthly_avg_temp[month]
        
        # Daily cycle: coolest at ~5AM, warmest at ~2PM
        daily_cycle = -5 * math.cos((hour - 14) * 2 * math.pi / 24)
        
        # Random variations
        random_component = noise * 2.5
        
        normal_value = round(monthly_base + daily_cycle + device_bias + random_component, 1)
    
    elif sensor_type == "humidity":
        # Seasonal component for humidity
        monthly_base = monthly_avg_humidity[month]
        
        # Daily cycle: highest in early morning, lowest in afternoon
        daily_cycle = 10 * math.cos((hour - 14) * 2 * math.pi / 24)
        
        # Random variations
        random_component = noise * 5
        
        # Ensure humidity stays in valid range
        humidity = monthly_base + daily_cycle + device_bias * 5 + random_component
        normal_value = round(min(max(humidity, 20), 100), 1)
    
    elif sensor_type == "pressure":
        # Base pressure around 1013 hPa with seasonal variations
        base_pressure = 1013
        seasonal_var = 5 * math.sin(day_of_year * 2 * math.pi / 365)
        
        # Random daily variations
        daily_var = 10 * np.random.normal(0, 0.5)
        
        normal_value = round(base_pressure + seasonal_var + daily_var + device_bias * 2, 1)
    
    elif sensor_type == "wind_speed":
        # Wind tends to be higher in winter months
        seasonal_factor = 1.0
        if month in [11, 12, 1, 2]:  # Winter months
            seasonal_factor = 1.5
        
        # Wind also tends to pick up during the day
        time_factor = 0.7 + 0.6 * math.sin((hour - 12) * math.pi / 12)
        
        base_speed = 5 * seasonal_factor * time_factor
        random_component = abs(noise) * 3  # Using abs to ensure non-negative
        
        normal_value = round(base_speed + random_component + device_bias, 1)
    
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
            normal_value = round(amount, 1)
        else:
            # No rain
            normal_value = 0.0
    
    elif sensor_type == "light_level":
        # Light level depends on time of day (daylight)
        if 7 <= hour <= 17:  # Daytime
            # Shorter daylight in winter
            if month in [11, 12, 1, 2]:
                if hour < 8 or hour > 16:
                    normal_value = round(random.uniform(0, 100), 1)
                else:
                    normal_value = round(random.uniform(500, 1000) + device_bias * 20, 1)
            else:
                normal_value = round(random.uniform(800, 1200) + device_bias * 20, 1)
        else:  # Night
            normal_value = round(random.uniform(0, 10), 1)
    else:
        normal_value = 0  # Default fallback
    
    # Check for anomalies
    anomaly = get_anomaly(timestamp, device_id, sensor_type)
    
    if anomaly:
        anomaly_type = anomaly["anomaly_type"]
        
        # Handle stuck value anomaly (maintain the same value across readings)
        if anomaly_type == "stuck_value":
            if anomaly["stuck_value"] is None:
                # First time seeing this stuck value anomaly, set the value
                anomaly["stuck_value"] = normal_value
            return anomaly["stuck_value"], anomaly
        
        # Handle gradual drift (accumulating over time)
        elif anomaly_type == "gradual_drift":
            if prev_anomaly and prev_anomaly["anomaly_type"] == "gradual_drift":
                # Continue the drift from previous value
                return AnomalyGenerator.gradual_drift(prev_value, anomaly["drift_factor"]), anomaly
            else:
                # Start the drift from normal value
                return AnomalyGenerator.gradual_drift(normal_value, anomaly["drift_factor"]), anomaly
        
        # Handle oscillation (alternating high/low)
        elif anomaly_type == "oscillation":
            # Toggle the oscillation state for next time
            anomaly["oscillation_state"] = not anomaly["oscillation_state"]
            return AnomalyGenerator.oscillation(normal_value, anomaly["oscillation_state"]), anomaly
        
        # Handle single-point anomalies
        elif anomaly_type == "sudden_spike":
            return AnomalyGenerator.sudden_spike(normal_value), anomaly
        elif anomaly_type == "sudden_drop":
            return AnomalyGenerator.sudden_drop(normal_value), anomaly
        elif anomaly_type == "high_noise":
            return AnomalyGenerator.high_noise(normal_value), anomaly
    
    # No anomaly case
    return normal_value, None

# Generate timestamps for the entire period
timestamps = []
current_time = start_date
while current_time <= end_date:
    timestamps.append(current_time)
    current_time += timedelta(minutes=interval_minutes)

# Keep track of previous values and anomalies for continuity
prev_values = {}
prev_anomalies = {}

# Generate the data
data = []
anomaly_records = []  # To keep track of when anomalies occur

for ts in timestamps:
    for device_id in device_ids:
        for sensor_type in sensor_types:
            # Use previous value and anomaly if available
            key = f"{device_id}_{sensor_type}"
            prev_value = prev_values.get(key, None)
            prev_anomaly = prev_anomalies.get(key, None)
            
            # Generate value with potential anomaly
            value, anomaly = generate_sensor_value(ts, sensor_type, device_id, prev_value, prev_anomaly)
            
            # Store current value and anomaly state for next iteration
            prev_values[key] = value
            prev_anomalies[key] = anomaly
            
            # Store the data
            data.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": device_id,
                "sensor_type": sensor_type,
                "value": value
            })
            
            # If there's an anomaly, record it
            if anomaly:
                anomaly_records.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "device_id": device_id,
                    "sensor_type": sensor_type,
                    "anomaly_type": anomaly["anomaly_type"],
                    "value": value
                })

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_filename = "prague_weather_iot_data_6months.csv"
df.to_csv(csv_filename, index=False)

# Save anomaly records for reference
anomaly_df = pd.DataFrame(anomaly_records)
anomaly_csv = "prague_weather_anomalies.csv"
anomaly_df.to_csv(anomaly_csv, index=False)

# Print summary info
total_records = len(df)
time_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
total_days = (end_date - start_date).days
device_count = len(device_ids)
sensor_count = len(sensor_types)
file_size_estimate = total_records * 100 / (1024 * 1024)  # Rough estimate in MB
total_anomalies = len(anomaly_records)
anomaly_percentage = total_anomalies / total_records * 100

print(f"Generated {csv_filename}")
print(f"Time period: {time_range} ({total_days} days)")
print(f"Records: {total_records:,}")
print(f"Devices: {device_count}")
print(f"Sensor types: {sensor_count}")
print(f"Measurement frequency: Every {interval_minutes} minutes")
print(f"Estimated file size: ~{file_size_estimate:.1f} MB")
print(f"Total anomalies: {total_anomalies} ({anomaly_percentage:.2f}% of data)")
print(f"Anomaly details saved to {anomaly_csv}")

# Show a few sample rows
print("\nSample data:")
print(df.head(10))

# Show some anomaly samples
if not anomaly_df.empty:
    print("\nSample anomalies:")
    print(anomaly_df.head(10))

# Print some statistics by sensor type for validation
print("\nSensor statistics (mean values by month):")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month'] = df['timestamp'].dt.month
monthly_stats = df.groupby(['month', 'sensor_type'])['value'].mean().unstack()
print(monthly_stats.round(2))