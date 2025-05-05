import streamlit as st
import pandas as pd
import sqlite3
from io import StringIO
import os
import re

# Set page configuration
st.set_page_config(
    page_title="SQL Query Simulator (Multiple CSVs)",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("SQL Query Simulator (Multiple CSVs)")
st.markdown("Upload or provide URLs for up to 5 CSV files to run SQL queries across multiple tables.")

# Function to sanitize table names
def sanitize_table_name(name):
    # Remove XX_ prefix pattern if it exists (where XX is a number)
    if re.match(r'^\d+_', name):
        name = re.sub(r'^\d+_', '', name)
    # Sanitize for SQL table name
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return sanitized

# Function to process a CSV URL
def process_csv_url(url, custom_name=None):
    try:
        # Convert GitHub URL to raw if needed
        if "github.com" in url and "/blob/" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            st.info(f"Converting to raw GitHub URL: {raw_url}")
            url = raw_url
            
        # Read CSV from URL with error handling
        try:
            df = pd.read_csv(url, error_bad_lines=False, warn_bad_lines=True)
        except:
            # For newer pandas versions (the parameter was renamed)
            df = pd.read_csv(url, on_bad_lines='skip')
        
        # Determine table name
        if custom_name:
            table_name = sanitize_table_name(custom_name)
        else:
            # Extract filename from URL 
            url_parts = url.split('/')
            csv_filename = url_parts[-1]
            # Get filename without extension
            filename = os.path.splitext(csv_filename)[0]
            table_name = sanitize_table_name(filename)
        
        return df, table_name
    except Exception as e:
        st.error(f"Error loading CSV from URL: {e}")
        st.error("Tips: For GitHub files, use the 'Raw' button on GitHub and copy that URL instead.")
        return None, None

# Function to get sample data
def get_sample_data(sample_type):
    if sample_type == "Sales Records":
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
        return df, "sample_sales"
    elif sample_type == "IoT Sensor Data":
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
2023-01-01T00:10:00,device002,humidity,49.5"""
        df = pd.read_csv(StringIO(iot_sample_data))
        return df, "iot_sensor_data"
    elif sample_type == "Customer Data":
        customer_data = """customer_id,name,email,registration_date,country
1001,John Smith,john.smith@example.com,2022-01-15,USA
1002,Emma Johnson,emma.j@example.com,2022-02-20,Canada
1003,Luis Rodriguez,luis.r@example.com,2022-01-30,Mexico
1004,Sarah Lee,slee@example.com,2022-03-10,USA
1005,Ahmed Hassan,a.hassan@example.com,2022-02-05,Egypt
1006,Anna Kowalski,anna.k@example.com,2022-03-15,Poland
1007,Hiroshi Tanaka,h.tanaka@example.com,2022-01-25,Japan
1008,Maria Garcia,m.garcia@example.com,2022-02-28,Spain
1009,Daniel Brown,d.brown@example.com,2022-03-05,UK
1010,Priya Patel,p.patel@example.com,2022-01-20,India"""
        df = pd.read_csv(StringIO(customer_data))
        return df, "customer_data"
    else:  # Orders data
        orders_data = """order_id,customer_id,order_date,total_amount,status
5001,1001,2023-02-10,125.99,Delivered
5002,1003,2023-02-12,89.50,Delivered
5003,1005,2023-02-15,210.75,Shipped
5004,1002,2023-02-18,45.25,Delivered
5005,1009,2023-02-20,150.00,Processing
5006,1007,2023-02-22,95.80,Shipped
5007,1010,2023-02-25,300.50,Processing
5008,1001,2023-02-28,75.25,Shipped
5009,1004,2023-03-02,180.00,Processing
5010,1008,2023-03-05,125.50,Processing"""
        df = pd.read_csv(StringIO(orders_data))
        return df, "orders_data"

# Sidebar for data source selection
with st.sidebar:
    st.header("Data Sources")
    st.markdown("Add up to 5 CSV sources")
    
    # Initialize data sources in session state if not already there
    if 'data_sources' not in st.session_state:
        st.session_state.data_sources = []
        for i in range(5):
            st.session_state.data_sources.append({
                'enabled': False,
                'type': 'URL to CSV',
                'url': '',
                'sample_type': 'Sales Records',
                'custom_name': f'table_{i+1}'
            })
    
    # Display controls for each potential data source
    for i in range(5):
        st.subheader(f"Data Source #{i+1}")
        st.session_state.data_sources[i]['enabled'] = st.checkbox(f"Enable Source #{i+1}", 
                                                                 value=st.session_state.data_sources[i]['enabled'])
        
        if st.session_state.data_sources[i]['enabled']:
            st.session_state.data_sources[i]['type'] = st.radio(f"Source type #{i+1}:", 
                                                             ["URL to CSV", "Sample Data"],
                                                             key=f"source_type_{i}")
            
            st.session_state.data_sources[i]['custom_name'] = st.text_input(f"Custom table name #{i+1}:", 
                                                                         value=st.session_state.data_sources[i]['custom_name'],
                                                                         key=f"custom_name_{i}")
            
            if st.session_state.data_sources[i]['type'] == "URL to CSV":
                st.session_state.data_sources[i]['url'] = st.text_input(f"CSV URL #{i+1}:", 
                                                                      value=st.session_state.data_sources[i]['url'],
                                                                      key=f"url_{i}")
                
            else:  # Sample data
                st.session_state.data_sources[i]['sample_type'] = st.selectbox(f"Sample data type #{i+1}:", 
                                                                           ["Sales Records", "IoT Sensor Data", "Customer Data", "Orders Data"],
                                                                           key=f"sample_type_{i}")

# Main content area
active_sources = [src for src in st.session_state.data_sources if src['enabled']]

if len(active_sources) > 0:
    # Create in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    
    # Data dictionary to store table information
    tables_info = {}
    
    # Create common_columns dictionary to track relationships
    common_columns = {}
    
    # Process each active data source
    for i, source in enumerate(active_sources):
        st.subheader(f"Data Source #{i+1}")
        
        if source['type'] == "URL to CSV" and source['url']:
            df, table_name = process_csv_url(source['url'], source['custom_name'])
            if df is not None:
                source['df'] = df
                source['table_name'] = table_name
                tables_info[table_name] = {
                    'columns': list(df.columns),
                    'rows': len(df),
                    'preview': df.head(5)
                }
                # Write to SQLite
                df.to_sql(table_name, conn, index=False, if_exists='replace')
                st.success(f"Loaded data into table '{table_name}' with {len(df)} rows and {len(df.columns)} columns")
        
        elif source['type'] == "Sample Data":
            df, default_table_name = get_sample_data(source['sample_type'])
            table_name = source['custom_name'] if source['custom_name'] else default_table_name
            table_name = sanitize_table_name(table_name)
            source['df'] = df
            source['table_name'] = table_name
            tables_info[table_name] = {
                'columns': list(df.columns),
                'rows': len(df),
                'preview': df.head(5)
            }
            # Write to SQLite
            df.to_sql(table_name, conn, index=False, if_exists='replace')
            st.success(f"Loaded sample data into table '{table_name}' with {len(df)} rows and {len(df.columns)} columns")
    
    # Display all available tables
    st.header("Available Tables for SQL Queries")
    for table_name, info in tables_info.items():
        with st.expander(f"{table_name} ({info['rows']} rows, {len(info['columns'])} columns)"):
            st.subheader("Columns")
            st.write(", ".join(info['columns']))
            st.subheader("Preview")
            st.dataframe(info['preview'])
    
    # SQL query section
    st.header("SQL Query")
    
    # Check if we have any tables loaded
    if not tables_info:
        st.warning("No tables were successfully loaded. Please check your data sources.")
        default_query = "-- No tables available. Please add valid data sources."
    # Generate a default query that joins tables if multiple tables exist
    elif len(tables_info) > 1:
        # Find potential join opportunities
        tables_list = list(tables_info.keys())
        common_columns = {}
        
        for i, table1 in enumerate(tables_list):
            for table2 in tables_list[i+1:]:
                common_cols = set(tables_info[table1]['columns']) & set(tables_info[table2]['columns'])
                if common_cols:
                    key = f"{table1}_{table2}"
                    common_columns[key] = list(common_cols)
        
        # Default query with join if possible
        if common_columns:
            # Simple default query for multiple tables without auto-join
            tables_list = list(tables_info.keys())
            default_query = "\n".join([f"-- Table: {table}" for table in tables_list])
            default_query += "\n\n"
            default_query += f"SELECT * FROM {tables_list[0]} LIMIT 10;"
        else:
            # No common columns, just show tables
            default_query = "\n".join([f"-- Table: {table}" for table in tables_list])
            default_query += "\n\n"
            default_query += f"SELECT * FROM {tables_list[0]} LIMIT 10;"
    else:
        # Only one table
        table_name = list(tables_info.keys())[0]
        default_query = f"SELECT * FROM {table_name} LIMIT 10;"
    
    query = st.text_area("Enter your SQL query", value=default_query, height=200)
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
    # Instructions when no sources are enabled
    st.info("Please enable at least one data source in the sidebar to get started.")
    
    # Display example app description
    st.header("How to Use This App")
    st.markdown("""
    1. Enable one or more data sources in the sidebar (up to 5)
    2. For each source, choose between URL to CSV or sample data
    3. Assign custom table names if desired
    4. Review the data previews and available tables
    5. Enter an SQL query or use one of the example queries
    6. Make sure your query ends with a semicolon (;)
    7. Click "Run Query" to execute
    8. View results and download as CSV if needed
    
    The app creates a temporary SQLite database in memory with all your data tables,
    allowing you to perform SQL operations across multiple tables including JOINs.
    """)

# Add footer
st.markdown("---")
st.markdown("Streamlit SQL Query Simulator - Multiple CSV to SQL App")