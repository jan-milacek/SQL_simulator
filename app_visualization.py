import streamlit as st
import pandas as pd
import sqlite3
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="IoT Sensor Data Visualization",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("IoT Sensor Data Visualization")
st.markdown("Analyze and visualize IoT sensor data using SQL queries and interactive charts")

# Function to load and preprocess data
def load_sensor_data(sensor_file, mereni_file):
    try:
        # Read CSV files
        sensors_df = pd.read_csv(sensor_file)
        mereni_df = pd.read_csv(mereni_file)
        
        # Convert timestamp to datetime
        try:
            mereni_df['casova_znamka'] = pd.to_datetime(mereni_df['casova_znamka'])
        except:
            st.warning("Could not convert 'casova_znamka' to datetime format. Using as-is.")
        
        # Convert installation date to datetime
        try:
            sensors_df['datum_instalace'] = pd.to_datetime(sensors_df['datum_instalace'])
        except:
            st.warning("Could not convert 'datum_instalace' to datetime format. Using as-is.")
            
        return sensors_df, mereni_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to create in-memory database with the data
def create_database(sensors_df, mereni_df):
    conn = sqlite3.connect(':memory:')
    sensors_df.to_sql('senzory', conn, index=False, if_exists='replace')
    mereni_df.to_sql('mereni', conn, index=False, if_exists='replace')
    return conn

# Function to suggest suitable visualization types for the data
def suggest_chart_types(df):
    suitable_charts = []
    
    # Check if we have timestamp and measurement value
    has_timestamp = any(col for col in df.columns if 'cas' in col.lower())
    has_value = any(col for col in df.columns if col in ['hodnota', 'value'])
    has_sensor_id = any(col for col in df.columns if col in ['sensor_id', 'senzor_id'])
    
    # Time series visualization
    if has_timestamp and has_value:
        suitable_charts.append('line')
    
    # Comparison between sensors
    if has_sensor_id and has_value:
        suitable_charts.append('bar')
        
    # Distribution of values
    if has_value:
        suitable_charts.append('histogram')
        suitable_charts.append('box')
    
    # If we have sensor type or location, we can use pie charts
    has_category = any(col for col in df.columns if col in ['typ', 'umisteni', 'jednotka'])
    if has_category and has_value:
        suitable_charts.append('pie')
    
    # For scatter plots, we need at least two numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 2:
        suitable_charts.append('scatter')
    
    # For heatmaps, we need a categorical column and a timestamp
    if has_timestamp and has_sensor_id and has_value:
        suitable_charts.append('heatmap')
    
    return suitable_charts

# Function to create visualizations
def create_visualization(df, chart_type, config):
    try:
        if chart_type == 'line':
            # Create a line chart for time series data
            fig = px.line(
                df, 
                x=config['x_axis'], 
                y=config['y_axis'],
                color=config.get('color_by'),
                title=config.get('title', 'Time Series Analysis'),
                labels={
                    config['x_axis']: config.get('x_label', config['x_axis']),
                    config['y_axis']: config.get('y_label', config['y_axis'])
                }
            )
            
        elif chart_type == 'bar':
            # Create a bar chart for comparing values across categories
            fig = px.bar(
                df, 
                x=config['x_axis'], 
                y=config['y_axis'],
                color=config.get('color_by'),
                title=config.get('title', 'Comparison Chart'),
                labels={
                    config['x_axis']: config.get('x_label', config['x_axis']),
                    config['y_axis']: config.get('y_label', config['y_axis'])
                }
            )
            
        elif chart_type == 'histogram':
            # Create a histogram for distribution analysis
            fig = px.histogram(
                df,
                x=config['x_axis'],
                color=config.get('color_by'),
                nbins=config.get('nbins', 20),
                title=config.get('title', 'Value Distribution'),
                labels={
                    config['x_axis']: config.get('x_label', config['x_axis'])
                }
            )
            
        elif chart_type == 'box':
            # Create a box plot for statistical distribution
            fig = px.box(
                df,
                x=config.get('group_by'),
                y=config['y_axis'],
                color=config.get('color_by'),
                title=config.get('title', 'Statistical Distribution'),
                labels={
                    config.get('group_by', ''): config.get('x_label', config.get('group_by', '')),
                    config['y_axis']: config.get('y_label', config['y_axis'])
                }
            )
            
        elif chart_type == 'pie':
            # Create a pie chart for showing proportions
            fig = px.pie(
                df,
                names=config['names'],
                values=config['values'],
                title=config.get('title', 'Proportion Analysis')
            )
            
        elif chart_type == 'scatter':
            # Create a scatter plot for correlation analysis
            fig = px.scatter(
                df,
                x=config['x_axis'],
                y=config['y_axis'],
                color=config.get('color_by'),
                size=config.get('size_by'),
                title=config.get('title', 'Correlation Analysis'),
                labels={
                    config['x_axis']: config.get('x_label', config['x_axis']),
                    config['y_axis']: config.get('y_label', config['y_axis'])
                }
            )
            
        elif chart_type == 'heatmap':
            # Create a heatmap for sensor data over time
            if all(col in config for col in ['pivot_index', 'pivot_columns', 'pivot_values']):
                # Create pivot table first
                pivot_df = df.pivot_table(
                    index=config['pivot_index'],
                    columns=config['pivot_columns'],
                    values=config['pivot_values'],
                    aggfunc=config.get('agg_func', 'mean')
                )
                
                fig = px.imshow(
                    pivot_df,
                    title=config.get('title', 'Heatmap Analysis'),
                    labels=dict(
                        x=config.get('x_label', config['pivot_columns']),
                        y=config.get('y_label', config['pivot_index']),
                        color=config.get('value_label', config['pivot_values'])
                    ),
                    color_continuous_scale='Viridis'
                )
            else:
                st.error("Missing configuration for heatmap visualization")
                return None
                
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return None
            
        # Apply common figure configurations
        fig.update_layout(
            height=600,
            width=800
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

# Main UI layout
st.sidebar.header("Data Sources")

# File upload section
with st.sidebar.expander("Upload Data Files", expanded=True):
    # Upload the sensors file
    uploaded_sensors = st.file_uploader("Upload sensors data (01_senzory.csv)", type=["csv"])
    
    # Upload the measurements file
    uploaded_mereni = st.file_uploader("Upload measurements data (01_mereni.csv)", type=["csv"])

# URL input section
with st.sidebar.expander("Or Provide URLs"):
    sensors_url = st.text_input("Sensors data URL (01_senzory.csv)")
    mereni_url = st.text_input("Measurements data URL (01_mereni.csv)")

# Main content area
if (uploaded_sensors and uploaded_mereni) or (sensors_url and mereni_url):
    # Load data based on input method
    if uploaded_sensors and uploaded_mereni:
        sensors_df = pd.read_csv(uploaded_sensors)
        mereni_df = pd.read_csv(uploaded_mereni)
    else:
        # Convert GitHub URLs to raw if needed
        if "github.com" in sensors_url and "/blob/" in sensors_url:
            sensors_url = sensors_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        if "github.com" in mereni_url and "/blob/" in mereni_url:
            mereni_url = mereni_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        
        try:
            sensors_df = pd.read_csv(sensors_url)
            mereni_df = pd.read_csv(mereni_url)
        except Exception as e:
            st.error(f"Error loading data from URLs: {e}")
            st.stop()
    
    # Preprocess data
    try:
        # Convert timestamp to datetime
        mereni_df['casova_znamka'] = pd.to_datetime(mereni_df['casova_znamka'])
        # Convert installation date to datetime
        sensors_df['datum_instalace'] = pd.to_datetime(sensors_df['datum_instalace'])
    except Exception as e:
        st.warning(f"Warning during datetime conversion: {e}")
    
    # Create database connection
    conn = create_database(sensors_df, mereni_df)
    
    # Create joined dataframe for visualization
    merged_df = pd.merge(
        mereni_df,
        sensors_df,
        on='sensor_id',
        how='left'
    )
    
    # Display dataset overview
    st.header("Dataset Overview")
    
    # Create tabs for data preview
    tab1, tab2, tab3 = st.tabs(["Sensors", "Measurements", "Joined Data"])
    
    with tab1:
        st.subheader("Sensors Data")
        st.dataframe(sensors_df, use_container_width=True)
        st.write(f"Total sensors: {len(sensors_df)}")
        
        # Show sensor types distribution
        if 'typ' in sensors_df.columns:
            type_counts = sensors_df['typ'].value_counts().reset_index()
            type_counts.columns = ['Sensor Type', 'Count']
            
            fig = px.pie(
                type_counts,
                names='Sensor Type',
                values='Count',
                title='Sensor Types Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Measurements Data")
        st.dataframe(mereni_df.head(100), use_container_width=True)
        st.write(f"Total measurements: {len(mereni_df)}")
        
        # Show basic statistics for measurements
        if 'hodnota' in mereni_df.columns:
            st.subheader("Measurement Statistics")
            stats_df = mereni_df.groupby('sensor_id')['hodnota'].agg(['count', 'mean', 'min', 'max']).reset_index()
            stats_df.columns = ['Sensor ID', 'Count', 'Mean', 'Min', 'Max']
            st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        st.subheader("Joined Data Preview")
        st.dataframe(merged_df.head(100), use_container_width=True)
    
    # SQL Query section
    st.header("SQL Query")
    
    # Generate example queries
    example_queries = {
        "Basic sensor list": "SELECT * FROM senzory;",
        "Recent measurements": "SELECT m.*, s.nazev_senzoru, s.typ, s.umisteni FROM mereni m JOIN senzory s ON m.sensor_id = s.sensor_id ORDER BY casova_znamka DESC LIMIT 100;",
        "Measurements by sensor type": "SELECT s.typ, COUNT(*) as count_measurements, AVG(m.hodnota) as avg_value FROM mereni m JOIN senzory s ON m.sensor_id = s.sensor_id GROUP BY s.typ;",
        "Measurements by location": "SELECT s.umisteni, COUNT(*) as count_measurements, AVG(m.hodnota) as avg_value FROM mereni m JOIN senzory s ON m.sensor_id = s.sensor_id GROUP BY s.umisteni;",
        "Daily aggregation": "SELECT date(m.casova_znamka) as day, s.nazev_senzoru, AVG(m.hodnota) as avg_value FROM mereni m JOIN senzory s ON m.sensor_id = s.sensor_id GROUP BY day, s.nazev_senzoru ORDER BY day;",
        "Min/Max values": "SELECT s.nazev_senzoru, MIN(m.hodnota) as min_value, MAX(m.hodnota) as max_value FROM mereni m JOIN senzory s ON m.sensor_id = s.sensor_id GROUP BY s.nazev_senzoru;"
    }
    
    # Let user select an example query
    selected_example = st.selectbox("Example queries:", list(example_queries.keys()))
    
    # Display the query in a text area
    query = st.text_area("SQL Query:", value=example_queries[selected_example], height=150)
    
    # Execute button
    if st.button("Execute Query"):
        try:
            # Run the query
            result_df = pd.read_sql_query(query, conn)
            
            # Display results
            st.subheader("Query Results")
            st.dataframe(result_df, use_container_width=True)
            
            # Download as CSV option
            st.download_button(
                label="Download Results as CSV",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name="query_results.csv",
                mime="text/csv"
            )
            
            # Visualization options based on the query results
            if not result_df.empty:
                st.header("Visualization")
                
                # Suggest chart types
                suitable_charts = suggest_chart_types(result_df)
                
                if not suitable_charts:
                    st.info("No suitable visualizations found for this query result. Try a query with temporal or numeric data.")
                else:
                    # Let user select a chart type
                    selected_chart = st.selectbox(
                        "Select chart type:",
                        suitable_charts,
                        format_func=lambda x: {
                            'line': 'Line Chart (Time Series)',
                            'bar': 'Bar Chart (Comparison)',
                            'histogram': 'Histogram (Distribution)',
                            'box': 'Box Plot (Statistical)',
                            'pie': 'Pie Chart (Proportion)',
                            'scatter': 'Scatter Plot (Correlation)',
                            'heatmap': 'Heatmap (Matrix View)'
                        }.get(x, x.capitalize())
                    )
                    
                    # Configure chart
                    st.subheader("Chart Configuration")
                    
                    # Initialize configuration
                    config = {}
                    
                    # Common configuration
                    config['title'] = st.text_input("Chart Title:", f"{selected_chart.capitalize()} Chart")
                    
                    # Configuration based on chart type
                    if selected_chart == 'line':
                        # For line chart, we need x-axis (typically time) and y-axis (value)
                        datetime_cols = [col for col in result_df.columns if result_df[col].dtype == 'datetime64[ns]']
                        if not datetime_cols:
                            # Also include string columns that might be dates
                            datetime_cols = [col for col in result_df.columns if 'date' in col.lower() or 'time' in col.lower() or 'cas' in col.lower()]
                        
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        
                        config['x_axis'] = st.selectbox("X-Axis (Time):", datetime_cols if datetime_cols else result_df.columns.tolist())
                        config['y_axis'] = st.selectbox("Y-Axis (Value):", numeric_cols if numeric_cols else result_df.columns.tolist())
                        
                        # Optional color grouping
                        categorical_cols = [col for col in result_df.columns if col not in [config['x_axis'], config['y_axis']]]
                        config['color_by'] = st.selectbox("Color by:", [None] + categorical_cols)
                        
                    elif selected_chart == 'bar':
                        # For bar chart, we need categories and values
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = [col for col in result_df.columns if col not in numeric_cols]
                        
                        config['x_axis'] = st.selectbox("X-Axis (Categories):", categorical_cols if categorical_cols else result_df.columns.tolist())
                        config['y_axis'] = st.selectbox("Y-Axis (Values):", numeric_cols if numeric_cols else result_df.columns.tolist())
                        
                        # Optional color grouping
                        remaining_cols = [col for col in result_df.columns if col not in [config['x_axis'], config['y_axis']]]
                        config['color_by'] = st.selectbox("Color by:", [None] + remaining_cols)
                        
                    elif selected_chart == 'histogram':
                        # For histogram, we need a numeric column
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        
                        config['x_axis'] = st.selectbox("Value Column:", numeric_cols if numeric_cols else result_df.columns.tolist())
                        config['nbins'] = st.slider("Number of Bins:", min_value=5, max_value=100, value=20)
                        
                        # Optional color grouping
                        categorical_cols = [col for col in result_df.columns if col not in numeric_cols]
                        config['color_by'] = st.selectbox("Color by:", [None] + categorical_cols)
                        
                    elif selected_chart == 'box':
                        # For box plot, we need a numeric column and optionally a grouping column
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = [col for col in result_df.columns if col not in numeric_cols]
                        
                        config['y_axis'] = st.selectbox("Value Column:", numeric_cols if numeric_cols else result_df.columns.tolist())
                        config['group_by'] = st.selectbox("Group by:", [None] + categorical_cols)
                        
                        # Optional color grouping
                        remaining_cols = [col for col in categorical_cols if col != config['group_by']]
                        config['color_by'] = st.selectbox("Color by:", [None] + remaining_cols)
                        
                    elif selected_chart == 'pie':
                        # For pie chart, we need categories and values
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = [col for col in result_df.columns if col not in numeric_cols]
                        
                        config['names'] = st.selectbox("Category Column:", categorical_cols if categorical_cols else result_df.columns.tolist())
                        config['values'] = st.selectbox("Value Column:", numeric_cols if numeric_cols else result_df.columns.tolist())
                        
                    elif selected_chart == 'scatter':
                        # For scatter plot, we need two numeric columns
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        
                        config['x_axis'] = st.selectbox("X-Axis:", numeric_cols if numeric_cols else result_df.columns.tolist())
                        config['y_axis'] = st.selectbox("Y-Axis:", [col for col in numeric_cols if col != config['x_axis']] if len(numeric_cols) > 1 else result_df.columns.tolist())
                        
                        # Optional color and size grouping
                        remaining_cols = [col for col in result_df.columns if col not in [config['x_axis'], config['y_axis']]]
                        config['color_by'] = st.selectbox("Color by:", [None] + remaining_cols)
                        
                        if len(numeric_cols) > 2:
                            size_options = [col for col in numeric_cols if col not in [config['x_axis'], config['y_axis']]]
                            config['size_by'] = st.selectbox("Size by:", [None] + size_options)
                        
                    elif selected_chart == 'heatmap':
                        # For heatmap, we need row index, column categories, and values
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = [col for col in result_df.columns if col not in numeric_cols]
                        datetime_cols = [col for col in result_df.columns if result_df[col].dtype == 'datetime64[ns]']
                        if not datetime_cols:
                            datetime_cols = [col for col in result_df.columns if 'date' in col.lower() or 'time' in col.lower() or 'cas' in col.lower()]
                        
                        config['pivot_index'] = st.selectbox("Row Dimension:", datetime_cols + categorical_cols if datetime_cols or categorical_cols else result_df.columns.tolist())
                        config['pivot_columns'] = st.selectbox("Column Dimension:", [col for col in categorical_cols if col != config['pivot_index']] if len(categorical_cols) > 1 else [col for col in result_df.columns if col != config['pivot_index']])
                        config['pivot_values'] = st.selectbox("Values:", numeric_cols if numeric_cols else [col for col in result_df.columns if col not in [config['pivot_index'], config['pivot_columns']]])
                        
                        config['agg_func'] = st.selectbox("Aggregation:", ['mean', 'median', 'sum', 'min', 'max', 'count'])
                    
                    # Create and display visualization
                    fig = create_visualization(result_df, selected_chart, config)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export options
                        st.subheader("Export Options")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Export as PNG"):
                                # Generate image bytes and provide download link
                                img_bytes = fig.to_image(format="png", engine="kaleido")
                                st.download_button(
                                    label="Download PNG",
                                    data=img_bytes,
                                    file_name=f"{selected_chart}_chart.png",
                                    mime="image/png"
                                )
                        with col2:
                            if st.button("Export as HTML"):
                                # Generate HTML and provide download link
                                html = fig.to_html(include_plotlyjs="cdn")
                                st.download_button(
                                    label="Download HTML",
                                    data=html,
                                    file_name=f"{selected_chart}_chart.html",
                                    mime="text/html"
                                )
        except Exception as e:
            st.error(f"Error executing query: {e}")
    
    # Add specialized visualization tabs for IoT data analysis
    st.header("Specialized IoT Visualizations")
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs([
        "Time Series Analysis", 
        "Sensor Comparison", 
        "Measurements Distribution",
        "Spatial Analysis"
    ])
    
    # Time Series Analysis Tab
    with viz_tabs[0]:
        st.subheader("Time Series Analysis")
        
        # Filter controls
        ts_col1, ts_col2 = st.columns(2)
        
        with ts_col1:
            # Select sensor(s)
            all_sensors = sensors_df['nazev_senzoru'].tolist()
            selected_sensors = st.multiselect(
                "Select sensors:", 
                all_sensors,
                default=[all_sensors[0]] if all_sensors else []
            )
        
        with ts_col2:
            # Time range selection
            if 'casova_znamka' in merged_df.columns:
                min_date = merged_df['casova_znamka'].min().date()
                max_date = merged_df['casova_znamka'].max().date()
                
                date_range = st.date_input(
                    "Date range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = min_date, max_date
            else:
                start_date, end_date = None, None
        
        # Time aggregation
        time_agg = st.selectbox(
            "Time Aggregation:",
            ["None", "Hourly", "Daily", "Weekly", "Monthly"],
        )
        
        # Generate visualization if sensors are selected
        if selected_sensors:
            # Filter data for selected sensors and time range
            filtered_df = merged_df[merged_df['nazev_senzoru'].isin(selected_sensors)]
            
            if start_date and end_date:
                filtered_df = filtered_df[
                    (filtered_df['casova_znamka'].dt.date >= start_date) & 
                    (filtered_df['casova_znamka'].dt.date <= end_date)
                ]
            
            # Apply time aggregation if selected
            if time_agg != "None" and 'casova_znamka' in filtered_df.columns:
                if time_agg == "Hourly":
                    filtered_df['time_group'] = filtered_df['casova_znamka'].dt.floor('H')
                elif time_agg == "Daily":
                    filtered_df['time_group'] = filtered_df['casova_znamka'].dt.floor('D')
                elif time_agg == "Weekly":
                    filtered_df['time_group'] = filtered_df['casova_znamka'].dt.to_period('W').dt.start_time
                elif time_agg == "Monthly":
                    filtered_df['time_group'] = filtered_df['casova_znamka'].dt.to_period('M').dt.start_time
                
                # Group by time and sensor
                agg_df = filtered_df.groupby(['time_group', 'nazev_senzoru'])['hodnota'].agg(['mean', 'min', 'max']).reset_index()
                
                # Rename columns
                agg_df.columns = ['casova_znamka', 'nazev_senzoru', 'mean_value', 'min_value', 'max_value']
                
                # Create plot with error bands
                fig = go.Figure()
                
                for sensor in selected_sensors:
                    sensor_df = agg_df[agg_df['nazev_senzoru'] == sensor]
                    
                    if not sensor_df.empty:
                        # Add main line (mean)
                        fig.add_trace(
                            go.Scatter(
                                x=sensor_df['casova_znamka'],
                                y=sensor_df['mean_value'],
                                mode='lines',
                                name=f"{sensor} (Mean)",
                                line=dict(width=2)
                            )
                        )
                        
                        # Add min/max range
                        fig.add_trace(
                            go.Scatter(
                                x=sensor_df['casova_znamka'].tolist() + sensor_df['casova_znamka'].tolist()[::-1],
                                y=sensor_df['max_value'].tolist() + sensor_df['min_value'].tolist()[::-1],
                                fill='toself',
                                fillcolor=f'rgba(0, 100, 80, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=False,
                                name=f"{sensor} (Range)",
                            )
                        )
            else:
                # Create time series plot without aggregation
                fig = px.line(
                    filtered_df,
                    x='casova_znamka',
                    y='hodnota',
                    color='nazev_senzoru',
                    title="Sensor Measurements Over Time",
                    labels={'casova_znamka': 'Time', 'hodnota': 'Value', 'nazev_senzoru': 'Sensor'}
                )
            
            # Update layout
            fig.update_layout(
                height=500,
                xaxis_title="Time",
                yaxis_title="Value"
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            st.subheader("Statistics for Selected Period")
            stats_df = filtered_df.groupby('nazev_senzoru')['hodnota'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).reset_index()
            
            # Format statistics for display
            stats_df.columns = ['Sensor', 'Count', 'Mean', 'Std Dev', 'Min', 'Max']
            st.dataframe(stats_df, use_container_width=True)
    
    # Sensor Comparison Tab
    with viz_tabs[1]:
        st.subheader("Sensor Comparison")
        
        # Comparison type selector
        comparison_type = st.radio(
            "Comparison Type:",
            ["Average Values", "Value Ranges", "Measurement Counts", "Custom Metric"],
            horizontal=True
        )
        
        # Get list of sensor types
        if 'typ' in sensors_df.columns:
            sensor_types = sensors_df['typ'].unique().tolist()
            
            # Choose sensors by type
            selected_type = st.selectbox("Filter by sensor type:", ["All Types"] + sensor_types)
            
            if selected_type != "All Types":
                # Get sensors of selected type
                type_sensors = sensors_df[sensors_df['typ'] == selected_type]['sensor_id'].tolist()
                comparison_df = merged_df[merged_df['sensor_id'].isin(type_sensors)]
            else:
                comparison_df = merged_df
        else:
            comparison_df = merged_df
        
        # Get sensor name list for the filtered data
        sensor_names = comparison_df['nazev_senzoru'].unique().tolist()
        
        # Create comparison visualization based on selected type
        if comparison_type == "Average Values":
            # Calculate average values by sensor
            avg_values = comparison_df.groupby('nazev_senzoru')['hodnota'].mean().reset_index()
            avg_values.columns = ['Sensor', 'Average Value']
            
            # Sort by average value
            avg_values = avg_values.sort_values('Average Value', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                avg_values,
                x='Sensor',
                y='Average Value',
                title="Average Measurement Value by Sensor",
                color='Average Value',
                color_continuous_scale='Viridis'
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
        elif comparison_type == "Value Ranges":
            # Calculate statistics by sensor
            range_values = comparison_df.groupby('nazev_senzoru')['hodnota'].agg(['min', 'mean', 'max']).reset_index()
            range_values.columns = ['Sensor', 'Min', 'Mean', 'Max']
            
            # Sort by mean value
            range_values = range_values.sort_values('Mean', ascending=False)
            
            # Create a range plot
            fig = go.Figure()
            
            for i, row in range_values.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Min'], row['Mean'], row['Max']],
                    y=[row['Sensor'], row['Sensor'], row['Sensor']],
                    mode='markers',
                    marker=dict(
                        color=['blue', 'green', 'red'],
                        size=[8, 12, 8],
                        symbol=['circle', 'diamond', 'circle']
                    ),
                    name=row['Sensor']
                ))
                
                # Add line connecting min to max
                fig.add_trace(go.Scatter(
                    x=[row['Min'], row['Max']],
                    y=[row['Sensor'], row['Sensor']],
                    mode='lines',
                    line=dict(width=2, color='gray'),
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title="Value Ranges by Sensor",
                xaxis_title="Measurement Value",
                yaxis_title="Sensor",
                height=max(400, len(range_values) * 30),
                showlegend=False
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
        elif comparison_type == "Measurement Counts":
            # Count measurements by sensor
            count_values = comparison_df.groupby('nazev_senzoru')['mereni_id'].count().reset_index()
            count_values.columns = ['Sensor', 'Measurement Count']
            
            # Sort by count
            count_values = count_values.sort_values('Measurement Count', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                count_values,
                x='Sensor',
                y='Measurement Count',
                title="Number of Measurements by Sensor",
                color='Measurement Count',
                color_continuous_scale='Viridis'
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
        elif comparison_type == "Custom Metric":
            # Let user select a metric
            metric_type = st.selectbox(
                "Select metric:",
                ["Coefficient of Variation", "Range (Max-Min)", "Measurement Frequency", "Standard Deviation"]
            )
            
            if metric_type == "Coefficient of Variation":
                # Calculate CV (std/mean)
                custom_values = comparison_df.groupby('nazev_senzoru')['hodnota'].agg(lambda x: x.std() / x.mean() * 100 if x.mean() != 0 else 0).reset_index()
                custom_values.columns = ['Sensor', 'Coefficient of Variation (%)']
                
                # Sort by CV
                custom_values = custom_values.sort_values('Coefficient of Variation (%)', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    custom_values,
                    x='Sensor',
                    y='Coefficient of Variation (%)',
                    title="Coefficient of Variation by Sensor",
                    color='Coefficient of Variation (%)',
                    color_continuous_scale='Viridis'
                )
                
            elif metric_type == "Range (Max-Min)":
                # Calculate range
                custom_values = comparison_df.groupby('nazev_senzoru')['hodnota'].agg(lambda x: x.max() - x.min()).reset_index()
                custom_values.columns = ['Sensor', 'Range (Max-Min)']
                
                # Sort by range
                custom_values = custom_values.sort_values('Range (Max-Min)', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    custom_values,
                    x='Sensor',
                    y='Range (Max-Min)',
                    title="Measurement Range by Sensor",
                    color='Range (Max-Min)',
                    color_continuous_scale='Viridis'
                )
                
            elif metric_type == "Measurement Frequency":
                # Calculate time difference between measurements
                if 'casova_znamka' in comparison_df.columns:
                    # Group by sensor and sort by time
                    comparison_df = comparison_df.sort_values(['nazev_senzoru', 'casova_znamka'])
                    
                    # Calculate frequency by sensor (measurements per day)
                    freq_values = []
                    
                    for sensor in sensor_names:
                        sensor_df = comparison_df[comparison_df['nazev_senzoru'] == sensor]
                        
                        if len(sensor_df) > 1:
                            # Calculate time range in days
                            time_range = (sensor_df['casova_znamka'].max() - sensor_df['casova_znamka'].min()).total_seconds() / (24 * 3600)
                            
                            # Calculate frequency (measurements per day)
                            frequency = len(sensor_df) / time_range if time_range > 0 else 0
                            
                            freq_values.append({
                                'Sensor': sensor,
                                'Measurements per Day': frequency
                            })
                    
                    if freq_values:
                        custom_values = pd.DataFrame(freq_values)
                        
                        # Sort by frequency
                        custom_values = custom_values.sort_values('Measurements per Day', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            custom_values,
                            x='Sensor',
                            y='Measurements per Day',
                            title="Measurement Frequency by Sensor",
                            color='Measurements per Day',
                            color_continuous_scale='Viridis'
                        )
                    else:
                        st.warning("Not enough time data to calculate frequency")
                        fig = None
                else:
                    st.warning("Timestamp data is required for frequency calculation")
                    fig = None
                
            elif metric_type == "Standard Deviation":
                # Calculate standard deviation
                custom_values = comparison_df.groupby('nazev_senzoru')['hodnota'].std().reset_index()
                custom_values.columns = ['Sensor', 'Standard Deviation']
                
                # Sort by std dev
                custom_values = custom_values.sort_values('Standard Deviation', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    custom_values,
                    x='Sensor',
                    y='Standard Deviation',
                    title="Measurement Standard Deviation by Sensor",
                    color='Standard Deviation',
                    color_continuous_scale='Viridis'
                )
            
            # Display chart if available
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Measurements Distribution Tab
    with viz_tabs[2]:
        st.subheader("Measurements Distribution")
        
        # Plot type selector
        dist_plot_type = st.radio(
            "Plot Type:",
            ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"],
            horizontal=True
        )
        
        # Distribution controls
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            # Select sensor(s)
            dist_sensors = st.multiselect(
                "Select sensors for distribution analysis:", 
                sensor_names,
                default=[sensor_names[0]] if sensor_names else []
            )
        
        with dist_col2:
            # Choose grouping
            if 'typ' in sensors_df.columns and 'umisteni' in sensors_df.columns:
                group_by = st.selectbox(
                    "Group by:",
                    ["None", "Sensor Type", "Location"]
                )
            elif 'typ' in sensors_df.columns:
                group_by = st.selectbox(
                    "Group by:",
                    ["None", "Sensor Type"]
                )
            elif 'umisteni' in sensors_df.columns:
                group_by = st.selectbox(
                    "Group by:",
                    ["None", "Location"]
                )
            else:
                group_by = "None"
        
        # Filter data for selected sensors
        if dist_sensors:
            dist_df = comparison_df[comparison_df['nazev_senzoru'].isin(dist_sensors)]
            
            # Apply grouping
            if group_by == "Sensor Type" and 'typ' in dist_df.columns:
                group_col = 'typ'
                chart_title = "Measurement Distribution by Sensor Type"
            elif group_by == "Location" and 'umisteni' in dist_df.columns:
                group_col = 'umisteni'
                chart_title = "Measurement Distribution by Location"
            else:
                group_col = 'nazev_senzoru'
                chart_title = "Measurement Distribution by Sensor"
            
            # Create distribution plot based on selected type
            if dist_plot_type == "Histogram":
                fig = px.histogram(
                    dist_df,
                    x='hodnota',
                    color=group_col,
                    title=chart_title,
                    labels={'hodnota': 'Measurement Value', group_col: group_col.capitalize()},
                    opacity=0.7,
                    barmode='overlay',
                    marginal="box"
                )
                
            elif dist_plot_type == "Box Plot":
                fig = px.box(
                    dist_df,
                    x=group_col,
                    y='hodnota',
                    color=group_col,
                    title=chart_title,
                    labels={'hodnota': 'Measurement Value', group_col: group_col.capitalize()},
                    points="all" if len(dist_df) < 100 else "outliers"
                )
                
            elif dist_plot_type == "Violin Plot":
                fig = px.violin(
                    dist_df,
                    x=group_col,
                    y='hodnota',
                    color=group_col,
                    title=chart_title,
                    labels={'hodnota': 'Measurement Value', group_col: group_col.capitalize()},
                    box=True,
                    points="all" if len(dist_df) < 100 else "outliers"
                )
                
            elif dist_plot_type == "KDE Plot":
                fig = px.density_contour(
                    dist_df,
                    x='hodnota',
                    color=group_col,
                    title=chart_title,
                    labels={'hodnota': 'Measurement Value', group_col: group_col.capitalize()},
                )
                
                # Add marginal distributions
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            
            # Update layout
            fig.update_layout(
                height=500
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display distribution statistics
            st.subheader("Distribution Statistics")
            
            # Calculate statistics by group
            if group_by != "None":
                dist_stats = dist_df.groupby(group_col)['hodnota'].describe().reset_index()
            else:
                dist_stats = dist_df.groupby('nazev_senzoru')['hodnota'].describe().reset_index()
            
            # Format statistics
            st.dataframe(dist_stats, use_container_width=True)
    
    # Spatial Analysis Tab
    with viz_tabs[3]:
        st.subheader("Spatial Analysis")
        
        # Check if location information is available
        if 'umisteni' in sensors_df.columns:
            # Get unique locations
            locations = sensors_df['umisteni'].unique().tolist()
            
            # Select analysis type
            spatial_type = st.radio(
                "Analysis Type:",
                ["Location Summary", "Measurement by Location", "Location Comparison"],
                horizontal=True
            )
            
            if spatial_type == "Location Summary":
                # Generate summary by location
                location_summary = []
                
                for loc in locations:
                    # Get sensors at this location
                    loc_sensors = sensors_df[sensors_df['umisteni'] == loc]['sensor_id'].tolist()
                    
                    # Get measurements for these sensors
                    loc_measurements = merged_df[merged_df['sensor_id'].isin(loc_sensors)]
                    
                    # Calculate summary statistics
                    if not loc_measurements.empty:
                        summary = {
                            'Location': loc,
                            'Sensors': len(loc_sensors),
                            'Measurements': len(loc_measurements),
                            'Avg Value': loc_measurements['hodnota'].mean(),
                            'Min Value': loc_measurements['hodnota'].min(),
                            'Max Value': loc_measurements['hodnota'].max()
                        }
                        
                        location_summary.append(summary)
                
                # Create summary dataframe
                if location_summary:
                    summary_df = pd.DataFrame(location_summary)
                    
                    # Display summary table
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Create visualization
                    fig = px.bar(
                        summary_df,
                        x='Location',
                        y='Sensors',
                        title="Number of Sensors by Location",
                        color='Sensors',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create measurement count visualization
                    fig = px.bar(
                        summary_df,
                        x='Location',
                        y='Measurements',
                        title="Number of Measurements by Location",
                        color='Measurements',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif spatial_type == "Measurement by Location":
                # Select location
                selected_location = st.selectbox("Select location:", locations)
                
                # Get sensors at this location
                loc_sensors = sensors_df[sensors_df['umisteni'] == selected_location]
                
                # Display sensors at this location
                st.subheader(f"Sensors at {selected_location}")
                st.dataframe(loc_sensors, use_container_width=True)
                
                # Get measurements for these sensors
                loc_measurements = merged_df[merged_df['sensor_id'].isin(loc_sensors['sensor_id'])]
                
                # Create time series visualization for this location
                if 'casova_znamka' in loc_measurements.columns:
                    fig = px.line(
                        loc_measurements,
                        x='casova_znamka',
                        y='hodnota',
                        color='nazev_senzoru',
                        title=f"Measurements at {selected_location} Over Time",
                        labels={'casova_znamka': 'Time', 'hodnota': 'Value', 'nazev_senzoru': 'Sensor'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add aggregation by sensor type if available
                    if 'typ' in loc_measurements.columns:
                        # Aggregate by sensor type
                        type_agg = loc_measurements.groupby(['typ', pd.Grouper(key='casova_znamka', freq='D')])['hodnota'].mean().reset_index()
                        
                        # Create aggregated visualization
                        fig = px.line(
                            type_agg,
                            x='casova_znamka',
                            y='hodnota',
                            color='typ',
                            title=f"Average Daily Measurements by Sensor Type at {selected_location}",
                            labels={'casova_znamka': 'Date', 'hodnota': 'Average Value', 'typ': 'Sensor Type'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
            elif spatial_type == "Location Comparison":
                # Select metric for comparison
                metric = st.selectbox(
                    "Comparison Metric:",
                    ["Average Value", "Value Range", "Measurement Count", "Sensor Count"]
                )
                
                # Prepare data for comparison
                location_data = []
                
                for loc in locations:
                    # Get sensors at this location
                    loc_sensors = sensors_df[sensors_df['umisteni'] == loc]['sensor_id'].tolist()
                    
                    # Get measurements for these sensors
                    loc_measurements = merged_df[merged_df['sensor_id'].isin(loc_sensors)]
                    
                    # Calculate selected metric
                    if not loc_measurements.empty:
                        if metric == "Average Value":
                            value = loc_measurements['hodnota'].mean()
                            metric_name = "Average Value"
                        elif metric == "Value Range":
                            value = loc_measurements['hodnota'].max() - loc_measurements['hodnota'].min()
                            metric_name = "Value Range"
                        elif metric == "Measurement Count":
                            value = len(loc_measurements)
                            metric_name = "Measurement Count"
                        elif metric == "Sensor Count":
                            value = len(loc_sensors)
                            metric_name = "Sensor Count"
                        
                        location_data.append({
                            'Location': loc,
                            metric_name: value
                        })
                
                # Create comparison dataframe
                if location_data:
                    comparison_df = pd.DataFrame(location_data)
                    
                    # Sort by selected metric
                    comparison_df = comparison_df.sort_values(metric_name, ascending=False)
                    
                    # Display comparison table
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Create comparison visualization
                    fig = px.bar(
                        comparison_df,
                        x='Location',
                        y=metric_name,
                        title=f"{metric_name} by Location",
                        color=metric_name,
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Location information is not available in the dataset. The 'umisteni' column is required for spatial analysis.")

# If no data is uploaded, show instructions
else:
    st.info("Please upload or provide URLs for your IoT sensor datasets to begin analysis.")
    
    # Instructions and example
    st.header("Expected Dataset Format")
    
    st.markdown("""
    This application expects two CSV files:
    
    1. **Sensor Metadata (01_senzory.csv)**
       - `sensor_id`: Unique identifier for each sensor
       - `nazev_senzoru`: Sensor name
       - `typ`: Sensor type
       - `umisteni`: Sensor location
       - `datum_instalace`: Installation date
    
    2. **Measurement Data (01_mereni.csv)**
       - `mereni_id`: Unique measurement ID
       - `sensor_id`: References the sensor from the sensors dataset
       - `casova_znamka`: Timestamp of the measurement
       - `hodnota`: Measured value
       - `jednotka`: Unit of measurement
    """)
    
    # Sample SQL queries
    st.header("Example SQL Queries")
    
    st.code("""
    -- Basic sensor list
    SELECT * FROM senzory;
    
    -- Recent measurements
    SELECT m.*, s.nazev_senzoru, s.typ, s.umisteni 
    FROM mereni m 
    JOIN senzory s ON m.sensor_id = s.sensor_id 
    ORDER BY casova_znamka DESC 
    LIMIT 100;
    
    -- Measurements by sensor type
    SELECT s.typ, COUNT(*) as count_measurements, AVG(m.hodnota) as avg_value 
    FROM mereni m 
    JOIN senzory s ON m.sensor_id = s.sensor_id 
    GROUP BY s.typ;
    
    -- Daily aggregation
    SELECT date(m.casova_znamka) as day, s.nazev_senzoru, AVG(m.hodnota) as avg_value 
    FROM mereni m 
    JOIN senzory s ON m.sensor_id = s.sensor_id 
    GROUP BY day, s.nazev_senzoru 
    ORDER BY day;
    """)

# Add footer
st.markdown("---")
st.markdown("IoT Sensor Data Visualization App | SQL Query-based Analysis Tool")