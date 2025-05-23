# SQL Query Simulator

A Streamlit application that allows users to run SQL queries on CSV data sources. This tool is perfect for data analysts, students learning SQL, and developers who want to quickly test SQL queries against CSV files without setting up a database.

## Try It Online

You can try the SQL Query Simulator online without installation:
[https://iot-sql-simulator.streamlit.app/](https://iot-sql-simulator.streamlit.app/)

![SQL Query Simulator Screenshot](https://raw.githubusercontent.com/jan-milacek/SQL_simulator/master/sql_simulator_screenshot.png)

## Features

- **Multiple Data Source Options**:
  - Upload CSV files from your computer
  - Link to CSV files via URL
  - Use built-in sample datasets (Sales Records or IoT Sensor Data)

- **Interactive SQL Environment**:
  - Write and execute custom SQL queries
  - Choose from pre-defined example queries
  - View query results in a data table format

- **Data Analysis Tools**:
  - Preview dataset and schema information
  - View statistics about query results
  - Download query results as CSV

- **IoT Data Analysis**:
  - Sample IoT dataset with timestamp, device_id, sensor_type, and value fields
  - Specialized example queries for time-series sensor data analysis

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. Clone this repository or download the source code:

```bash
git clone https://github.com/jan-milacek/SQL_simulator.git
cd SQL_simulator
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

Or install the packages individually:

```bash
pip install streamlit pandas
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Select a data source:
   - Upload a CSV file
   - Enter a URL to a CSV file
   - Choose one of the sample datasets

4. Explore the dataset preview and schema

5. Write your SQL query or select from example queries

6. Click "Run Query" to execute and view results



## Customization

You can modify the app.py file to:

- Add more sample datasets
- Customize the UI layout and styling
- Add additional SQL query examples
- Implement more data analysis functions

## How It Works

The application creates an in-memory SQLite database to store your CSV data, allowing you to run SQL queries without needing to set up an external database. This makes it perfect for quick data analysis and SQL learning.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [SQLite](https://www.sqlite.org/) for in-memory database operations
- Uses [Pandas](https://pandas.pydata.org/) for data manipulation