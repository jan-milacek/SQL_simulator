import streamlit as st
import pandas as pd
import sqlite3
from io import StringIO
import os
import re

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
        # Get the filename without extension
        filename = os.path.splitext(uploaded_file.name)[0]
        # Remove XX_ prefix pattern if it exists (where XX is a number)
        if re.match(r'^\d+_', filename):
            filename = re.sub(r'^\d+_', '', filename)
        # Sanitize for SQL table name
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
    elif data_source == "URL to CSV" and csv_url:
        try:
            # Convert GitHub URL to raw if needed
            if "github.com" in csv_url and "/blob/" in csv_url:
                raw_url = csv_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                st.info(f"Converting to raw GitHub URL: {raw_url}")
                csv_url = raw_url
                
            # Read CSV from URL with error handling
            try:
                df = pd.read_csv(csv_url, error_bad_lines=False, warn_bad_lines=True)
            except:
                # For newer pandas versions (the parameter was renamed)
                df = pd.read_csv(csv_url, on_bad_lines='skip')
            
            # Extract filename from URL 
            url_parts = csv_url.split('/')
            csv_filename = url_parts[-1]
            # Get filename without extension
            filename = os.path.splitext(csv_filename)[0]
            # Remove XX_ prefix pattern if it exists (where XX is a number)
            if re.match(r'^\d+_', filename):
                filename = re.sub(r'^\d+_', '', filename)
            # Sanitize for SQL table name
            table_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
            
            st.success(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
            st.info(f"Table name in SQL: {table_name}")
        except Exception as e:
            st.error(f"Error loading CSV from URL: {e}")
            st.error("Tips: For GitHub files, use the 'Raw' button on GitHub and copy that URL instead.")
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
    
    # Default query based on table type
    default_query = f"SELECT * FROM {table_name} LIMIT 10;"
    
    query = st.text_area("Enter your SQL query", value=default_query, height=150)
    st.info("Make sure your query ends with a semicolon (;)")
    
    # Execute the query
    if st.button("Run Query"):
        try:
            # Check if the query ends with a semicolon
            if not query.strip().endswith(';'):
                st.error("Query must end with a semicolon (;)")
            else:
                query_result = pd.read_sql_query(query.strip(), conn)
                
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
    5. Make sure your query ends with a semicolon (;)
    6. Click "Run Query" to execute
    7. View results and download as CSV if needed
    
    The app creates a temporary SQLite database in memory with your CSV data.
    """)

# Add footer
st.markdown("---")
st.markdown("Streamlit SQL Query Simulator - CSV to SQL App")