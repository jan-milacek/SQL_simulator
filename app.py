import streamlit as st
import pandas as pd
import sqlite3
from io import StringIO
import os

# Set page configuration
st.set_page_config(
    page_title="SQL Query Simulator",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("SQL Query Simulator")
st.markdown("Upload a CSV file, provide a URL, or use sample data to run SQL queries.")

# Sidebar for data source selection
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Choose a data source:", ["Upload CSV", "URL to CSV", "Sample Data"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    elif data_source == "URL to CSV":
        csv_url = st.text_input("Enter URL to CSV file:", "")
        st.info("Enter a direct URL to a CSV file")
    else:  # Sample data
        sample_type = st.selectbox("Sample data type:", ["Sales Records", "IoT Sensor Data"])
        st.info(f"Using sample data: {sample_type}")

# Main content area
if data_source == "Upload CSV" and uploaded_file is not None or data_source == "URL to CSV" and csv_url or data_source == "Sample Data":
    # Load the data
    if data_source == "Upload CSV" and uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        table_name = os.path.splitext(uploaded_file.name)[0]
    elif data_source == "URL to CSV" and csv_url:
        try:
            # Read CSV from URL
            df = pd.read_csv(csv_url)
            table_name = "url_data"
        except Exception as e:
            st.error(f"Error loading CSV from URL: {e}")
            st.stop()
    else:
        # Create sample data based on selection
        if data_source == "Sample Data" and sample_type == "Sales Records":
            sample_data = """date,product,category,price,quantity,revenue
2023-01-01,Laptop,Electronics,1200,5,6000
2023-01-02,Smartphone,Electronics,800,10,8000
2023-01-03,Headphones,Electronics,200,20,4000
2023-01-04,T-shirt,Clothing,25,40,1000
2023-01-05,Jeans,Clothing,50,15,750
2023-01-06,Sneakers,Footwear,80,12,960
2023-01-07,Sandals,Footwear,30,25,750
2023-01-08,Coffee Maker,Appliances,120,8,960
2023-01-09,Toaster,Appliances,45,14,630
2023-01-10,Blender,Appliances,70,6,420"""
            df = pd.read_csv(StringIO(sample_data))
            table_name = "sample_sales"
        else:  # IoT Sensor Data
            iot_sample_data = """timestamp,device_id,sensor_type,value
2023-01-01T00:00:00,device001,temperature,22.5
2023-01-01T00:05:00,device001,temperature,22.8
2023-01-01T00:10:00,device001,temperature,23.1
2023-01-01T00:00:00,device001,humidity,45.2
2023-01-01T00:05:00,device001,humidity,45.5
2023-01-01T00:10:00,device001,humidity,46.0
2023-01-01T00:00:00,device002,temperature,21.4
2023-01-01T00:05:00,device002,temperature,21.5
2023-01-01T00:10:00,device002,temperature,21.7
2023-01-01T00:00:00,device002,humidity,48.9
2023-01-01T00:05:00,device002,humidity,49.1
2023-01-01T00:10:00,device002,humidity,49.5
2023-01-01T00:00:00,device003,temperature,24.2
2023-01-01T00:05:00,device003,temperature,24.5
2023-01-01T00:10:00,device003,temperature,24.7
2023-01-01T00:00:00,device003,humidity,42.3
2023-01-01T00:05:00,device003,humidity,42.1
2023-01-01T00:10:00,device003,humidity,42.5
2023-01-01T00:00:00,device001,pressure,1013.2
2023-01-01T00:05:00,device001,pressure,1013.4
2023-01-01T00:10:00,device001,pressure,1013.3
2023-01-01T00:00:00,device002,pressure,1012.8
2023-01-01T00:05:00,device002,pressure,1012.9
2023-01-01T00:10:00,device002,pressure,1013.0"""
            df = pd.read_csv(StringIO(iot_sample_data))
            table_name = "iot_sensor_data"
    
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    
    # Write the dataframe to the database
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    # Display the dataset
    st.header("Dataset Preview")
    st.dataframe(df.head(10))
    
    # Display table schema
    st.header("Table Schema")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Values': [df[col].count() for col in df.columns],
        'Null Values': [df[col].isna().sum() for col in df.columns]
    })
    st.dataframe(col_info)
    
    # SQL query section
    st.header("SQL Query")
    
    # Provide some example queries based on data type
    if table_name == "iot_sensor_data":
        examples = {
            "Select all data": f"SELECT * FROM {table_name}",
            "Group by device": f"SELECT device_id, sensor_type, AVG(value) as avg_value FROM {table_name} GROUP BY device_id, sensor_type",
            "Filter by sensor type": f"SELECT * FROM {table_name} WHERE sensor_type = 'temperature'",
            "Time-based analysis": f"SELECT strftime('%H:%M', timestamp) as time, AVG(value) as avg_value FROM {table_name} WHERE sensor_type = 'humidity' GROUP BY time",
            "Compare devices": f"SELECT device_id, MAX(value) as max_temp FROM {table_name} WHERE sensor_type = 'temperature' GROUP BY device_id ORDER BY max_temp DESC"
        }
    else:
        examples = {
            "Select all data": f"SELECT * FROM {table_name}",
            "Group by category": f"SELECT category, SUM(revenue) as total_revenue FROM {table_name} GROUP BY category ORDER BY total_revenue DESC",
            "Filter by date": f"SELECT * FROM {table_name} WHERE date >= '2023-01-05'",
            "Calculate average": f"SELECT product, AVG(price) as avg_price FROM {table_name} GROUP BY product"
        }
    
    selected_example = st.selectbox("Example queries", ["Custom"] + list(examples.keys()))
    
    if selected_example == "Custom":
        default_query = f"SELECT * FROM {table_name} LIMIT 10"
    else:
        default_query = examples[selected_example]
    
    query = st.text_area("Enter your SQL query", value=default_query, height=150)
    
    # Execute the query
    if st.button("Run Query"):
        try:
            query_result = pd.read_sql_query(query, conn)
            
            # Display query results
            st.header("Query Results")
            st.dataframe(query_result)
            
            # Download results button
            csv = query_result.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv"
            )
            
            # Show some statistics about the results
            st.header("Result Statistics")
            st.write(f"Number of rows: {len(query_result)}")
            st.write(f"Number of columns: {len(query_result.columns)}")
            
            # For numerical columns, show basic statistics
            numeric_cols = query_result.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.subheader("Numerical Column Statistics")
                st.dataframe(query_result[numeric_cols].describe())
            
        except Exception as e:
            st.error(f"Error executing query: {e}")
    
    # Close the database connection when the app is done
    conn.close()

else:
    # Instructions when no file is uploaded
    st.info("Please upload a CSV file or use the sample data to get started.")
    
    # Display example app description
    st.header("How to Use This App")
    st.markdown("""
    1. Upload a CSV file using the sidebar
    2. Alternatively, check "Use sample data" to test with example data
    3. Review the data preview and schema
    4. Enter an SQL query or select from example queries
    5. Click "Run Query" to execute
    6. View results and download as CSV if needed
    
    **Example Queries You Can Run:**
    - Basic select: `SELECT * FROM table_name LIMIT 10`
    - Filtering: `SELECT * FROM table_name WHERE column_name > value`
    - Grouping: `SELECT column1, SUM(column2) FROM table_name GROUP BY column1`
    - Sorting: `SELECT * FROM table_name ORDER BY column_name DESC`
    
    The app creates a temporary SQLite database in memory with your CSV data.
    """)

# Add footer
st.markdown("---")
st.markdown("Streamlit SQL Query Simulator - CSV to SQL App")