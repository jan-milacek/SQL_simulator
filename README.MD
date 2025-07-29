# SQL Simulator for Educational Data Analysis

**Python/Streamlit-based interactive SQL learning platform designed for high school education**

A comprehensive SQL simulator built with Python and Streamlit that provides hands-on learning experience for students analyzing real datasets. Originally developed for a 9-week high school course teaching SQL fundamentals through practical data analysis, with specialized features for IoT sensor data visualization.

## 🎯 Purpose

This educational tool bridges the gap between theoretical SQL learning and practical data analysis by providing:
- **Interactive SQL environment** - Web-based Streamlit interface with real-time query execution
- **Multiple CSV support** - Load up to 5 CSV files simultaneously for complex analysis
- **Real IoT datasets** - Specialized visualization tools for sensor data analysis
- **Progressive learning** - From basic SELECT statements to complex JOINs and aggregations
- **Immediate feedback** - Real-time query execution with error handling and explanations
- **Educational focus** - Designed specifically for classroom learning with built-in examples

## ✨ Features

### Core SQL Simulator (`app.py`)
- **Multi-table support** - Upload up to 5 CSV files as database tables
- **Smart table naming** - Automatic sanitization of CSV filenames to valid SQL table names
- **In-memory SQLite** - Fast, temporary database creation for educational use
- **Sample datasets** - Built-in examples (Sales, IoT, Customer, Orders data)
- **GitHub integration** - Direct CSV loading from GitHub repositories
- **Query validation** - Syntax checking and helpful error messages
- **Results export** - Download query results as CSV files

### Advanced IoT Data Visualization (`app_visualization.py`)
- **Specialized IoT analysis** - Purpose-built for sensor data workflows
- **Time series visualization** - Interactive charts for temporal data analysis
- **Sensor comparison tools** - Side-by-side analysis of multiple sensors
- **Distribution analysis** - Histograms, box plots, and statistical summaries
- **Spatial analysis** - Location-based sensor performance analysis
- **Custom chart configuration** - User-configurable visualization parameters
- **Multiple chart types** - Line, bar, histogram, box, pie, scatter, and heatmap visualizations

### Educational Features
- **Example queries** - Pre-built SQL queries demonstrating key concepts
- **Progressive difficulty** - From basic SELECT to complex analytical queries
- **Sample datasets** - Realistic data scenarios for hands-on practice
- **Data exploration tools** - Interactive data preview and column analysis
- **Error-friendly environment** - Safe learning space with helpful error messages

## 🏗️ Architecture

```
SQL Simulator
├── app.py                      # Multi-CSV SQL query simulator
├── app_visualization.py        # IoT-focused data visualization tool
├── requirements.txt           # Python dependencies
└── README.md                  # This file

Tech Stack:
├── Streamlit                  # Web application framework
├── SQLite                     # In-memory database engine
├── Pandas                     # Data manipulation and analysis
├── Plotly                     # Interactive visualization
└── Python 3.7+              # Core programming language
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/jan-milacek/SQL_simulator.git
cd SQL_simulator
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

#### Basic SQL Simulator
```bash
streamlit run app.py
```
- Upload CSV files or use sample data
- Write SQL queries across multiple tables
- Execute and download results

#### IoT Data Visualization Tool
```bash
streamlit run app_visualization.py
```
- Upload IoT sensor datasets (sensors + measurements)
- Perform SQL analysis with visualization
- Create interactive dashboards

### Expected Data Format (IoT Tool)

The IoT visualization tool expects two CSV files:

**Sensors Data (`01_senzory.csv`)**
```csv
sensor_id,nazev_senzoru,typ,umisteni,datum_instalace
1,Temperature Sensor 1,temperature,Room A,2023-01-01
2,Humidity Sensor 1,humidity,Room A,2023-01-01
```

**Measurements Data (`01_mereni.csv`)**
```csv
mereni_id,sensor_id,casova_znamka,hodnota,jednotka
1,1,2023-01-01 10:00:00,22.5,°C
2,1,2023-01-01 10:05:00,22.8,°C
```

## 📊 Sample Use Cases

### Educational SQL Learning
```sql
-- Basic sensor exploration
SELECT * FROM senzory WHERE typ = 'temperature';

-- Time series analysis
SELECT DATE(casova_znamka) as day, 
       AVG(hodnota) as avg_temp
FROM mereni m 
JOIN senzory s ON m.sensor_id = s.sensor_id 
WHERE s.typ = 'temperature'
GROUP BY day
ORDER BY day;

-- Multi-table analysis
SELECT s.umisteni, 
       COUNT(*) as measurement_count,
       AVG(m.hodnota) as avg_value
FROM mereni m 
JOIN senzory s ON m.sensor_id = s.sensor_id 
GROUP BY s.umisteni;
```

### Data Visualization Examples
- **Time Series**: Sensor readings over time with trend analysis
- **Comparison**: Performance differences between sensor locations
- **Distribution**: Statistical analysis of measurement patterns
- **Spatial**: Geographic distribution of sensor performance

## 🎓 Educational Implementation

### 9-Week Course Structure
This simulator was designed for a comprehensive SQL curriculum:

- **Week 1-2:** Basic SELECT, WHERE, ORDER BY using sample data
- **Week 3:** Aggregation functions (COUNT, AVG, SUM) with IoT datasets
- **Week 4:** JOINs across sensor and measurement tables
- **Week 5:** Time-series analysis with DATE functions
- **Week 6-7:** Data visualization and interactive chart creation
- **Week 8-9:** Advanced analytics and dashboard development

### Learning Outcomes
Students completing the course can:
- Write complex SQL queries for real-world data analysis
- Understand time-series data patterns and seasonality
- Create meaningful visualizations from raw sensor data
- Perform statistical analysis using SQL aggregate functions
- Build interactive dashboards for data presentation
- Connect SQL analysis to business insights and decision-making

## 🛠️ Technical Details

### Supported SQL Features
- **Complete SELECT syntax** - WHERE, ORDER BY, GROUP BY, HAVING
- **Aggregate functions** - COUNT, SUM, AVG, MIN, MAX
- **JOIN operations** - INNER, LEFT, RIGHT, FULL OUTER
- **Date/time functions** - DATE, TIME, datetime manipulation
- **Window functions** - ROW_NUMBER, RANK, LAG, LEAD
- **Common Table Expressions** - WITH clause for complex queries

### Visualization Capabilities
- **Interactive charts** using Plotly.js
- **Real-time data updates** as queries change
- **Export functionality** - PNG, HTML, and CSV formats
- **Responsive design** - Works on desktop and tablet devices
- **Chart customization** - Colors, labels, and styling options

### Performance Characteristics
- **Fast query execution** - In-memory SQLite for educational datasets
- **Efficient visualization** - Optimized for datasets up to 100K rows
- **Low resource usage** - Suitable for classroom computer labs
- **Browser-based** - No local installation required for students

## 🤝 Contributing

This project was created for educational purposes and welcomes contributions that enhance the learning experience:

### Areas for Contribution
- **New sample datasets** representing different industries and use cases
- **Additional visualization types** for specialized data analysis
- **Enhanced error messages** to guide student learning
- **Performance improvements** for larger educational datasets
- **Documentation improvements** for classroom implementation
- **Internationalization** for non-English speaking classrooms

### Development Setup
1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes and test thoroughly
6. Submit a pull request with clear description

## 📝 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
sqlite3 (built-in)
plotly>=5.15.0
numpy>=1.21.0
```

## 📞 Support & Usage

### For Educators
- **Classroom deployment** - Single-command startup for computer labs
- **Curriculum integration** - Structured progression from basics to advanced
- **Assessment tools** - Query validation and result checking
- **No setup required** - Students access via web browser

### For Students
- **Guided learning** - Example queries and progressive difficulty
- **Safe environment** - No risk of damaging production systems
- **Immediate feedback** - Real-time query results and error explanations
- **Portfolio building** - Downloadable results for project documentation

### Technical Support
For questions about educational use or technical implementation:
- Create an issue in this repository
- Contact: jan.milacek@gmail.com
- Check the Wiki for common solutions

## 🔗 Related Projects

- [MQTT IoT Simulator](../mqtt_dashboard) - Companion tool for real-time sensor data simulation
- [Course Materials](../course-materials) - Complete 9-week curriculum resources

## 📜 License

MIT License - Feel free to use, modify, and distribute for educational purposes.

## 🎯 Real-World Impact

### Educational Success Metrics
- **High school students** successfully created professional-grade data analysis
- **100% completion rate** in pilot implementation
- **Strong foundation building** for advanced analytics and data science careers
- **Industry-relevant skills** using real SQL syntax and best practices

### Why This Approach Works
- **Hands-on learning** beats theoretical instruction for technical skills
- **Real datasets** provide context that textbook examples cannot
- **Immediate feedback** accelerates the learning process
- **Progressive complexity** builds confidence while maintaining challenge
- **Visual results** help students understand the power of data analysis

---

**Built with ❤️ for data education in Czech Republic**

*This tool demonstrates that complex technical concepts can be made accessible through thoughtful design, practical application, and student-centered pedagogy. The combination of SQL fundamentals with modern visualization creates a powerful learning experience that prepares students for real-world data analysis challenges.*
- **Real IoT datasets** - Temperature

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No additional software installation required

### Quick Start
1. Clone the repository:
```bash
git clone https://github.com/jan-milacek/sql-simulator.git
cd sql-simulator
```

2. Start the local server:
```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server

# Using PHP
php -S localhost:8000
```

3. Open browser and navigate to `http://localhost:8000`

4. Begin with the guided tutorial or explore sample datasets

## 📊 Sample Datasets

### IoT Sensor Data
- **Temperature readings** - Indoor/outdoor sensors with timestamps
- **Humidity measurements** - Multi-location environmental monitoring
- **Energy consumption** - Smart meter data with usage patterns
- **Air quality sensors** - CO2, PM2.5, and other environmental factors

### Example Queries
```sql
-- Basic sensor data exploration
SELECT sensor_id, temperature, humidity, timestamp 
FROM sensor_readings 
WHERE timestamp > '2024-01-01' 
ORDER BY timestamp DESC;

-- Daily temperature averages
SELECT 
    DATE(timestamp) as date,
    AVG(temperature) as avg_temp,
    MIN(temperature) as min_temp,
    MAX(temperature) as max_temp
FROM sensor_readings 
GROUP BY DATE(timestamp)
ORDER BY date;

-- Sensor performance analysis
SELECT 
    sensor_id,
    COUNT(*) as reading_count,
    AVG(temperature) as avg_temp,
    STDDEV(temperature) as temp_variance
FROM sensor_readings 
GROUP BY sensor_id
HAVING COUNT(*) > 100;
```

## 🎓 Educational Use

### Course Integration
This simulator was designed for a 9-week SQL curriculum:
- **Week 1-2:** Basic SELECT, WHERE, ORDER BY
- **Week 3:** Aggregation functions and GROUP BY
- **Week 4:** JOINs across sensor tables
- **Week 5:** Time-series analysis and date functions
- **Week 6-7:** Data visualization and reporting
- **Week 8-9:** Advanced analytics and dashboard creation

### Learning Outcomes
Students successfully completing the curriculum can:
- Write complex SQL queries for data analysis
- Understand time-series data patterns
- Create meaningful visualizations from raw data
- Connect SQL analysis to business insights
- Build foundation for advanced analytics tools

## 🛠️ Technical Details

### Supported SQL Features
- **SELECT statements** with all standard clauses
- **Aggregate functions** (COUNT, SUM, AVG, MIN, MAX, STDDEV)
- **JOIN operations** (INNER, LEFT, RIGHT, FULL OUTER)
- **Window functions** for advanced analytics
- **Date/time functions** for time-series analysis
- **Common Table Expressions (CTEs)** for complex queries

### Browser Compatibility
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

### Performance
- Handles datasets up to 100,000 rows efficiently
- Query execution typically under 1 second
- Optimized for educational datasets and common query patterns

## 🤝 Contributing

This project was created for educational purposes and welcomes contributions that enhance the learning experience:

- **New datasets** representing different IoT scenarios
- **Additional exercises** for various skill levels
- **Visualization improvements** for better data understanding
- **Documentation enhancements** for classroom use
- **Bug fixes** and performance optimizations

## 📝 License

MIT License - Feel free to use, modify, and distribute for educational purposes.

## 🎯 Use Cases

### For Educators
- **Classroom demonstrations** with immediate visual feedback
- **Assignment creation** with built-in datasets
- **Progress tracking** through query complexity
- **No setup required** for students or IT departments

### For Students
- **Hands-on practice** with realistic data scenarios
- **Safe learning environment** with no risk of breaking systems
- **Immediate feedback** on query syntax and logic
- **Portfolio building** with query examples and results

### For Self-Learners
- **Structured curriculum** with progressive difficulty
- **Real-world datasets** for practical experience
- **Comprehensive examples** demonstrating best practices
- **Foundation building** for advanced analytics careers


## 📞 Support

For questions about educational use or technical implementation:
- Create an issue in this repository
- Contact: jan.milacek@gmail.com

---

**Built with ❤️ for data education**

*This tool demonstrates that complex technical concepts can be made accessible through thoughtful design and practical application.*
