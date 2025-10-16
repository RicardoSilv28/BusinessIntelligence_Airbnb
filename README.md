# BusinessIntelligence_Airbnb

**End-to-end BI solution analyzing Airbnb Lisbon data using Tableau, Pentaho, PostgreSQL, and Python for predictive insights.**

---

## 🧭 Overview
This project is a comprehensive **Business Intelligence (BI)** solution developed as part of the *Business Intelligence* course at the **University of Coimbra (FCTUC)**.  
It explores the **Airbnb market in Lisbon (and partially Porto)** through ETL processing, data warehousing, visualization with Tableau, and predictive analytics using Python.

**Authors:**
- Ricardo Santiago  
- Ricardo Silva  

---

## 📁 Project Structure

```
📦 BusinessIntelligence_Airbnb
├── 📁 Tableau
│   ├── Screenshots of Tableau visualizations (Part 1)
│   └── Public link to the interactive Tableau dashboard
│
├── 📁 Machine Learning
│   ├── Python script (Part 2)
│   ├──── Classification models, and time-series forecasting
│   └──── GUI prototype for price recommendation
│
├── 📁 Presentations
│   ├── Slides used for project presentations (Parts 1 & 2)
│
├── 📄 BI_Report_Submission_Part1and2_PT.pdf
│   └── Full project report (in Portuguese)
│
└── 📄 Project_Description.pdf
    └── Official project brief and academic requirements
```

---

## 🧩 Part 1 – Business Intelligence with Tableau

### 🎯 Objective
Design and implement a **Data Warehouse (DW)** and **OLAP system** to analyze Airbnb listings in Lisbon.  
The goal was to extract insights on host performance, pricing strategies, and geographic trends to support data-driven decisions.

### 🛠️ Technologies Used
- **PostgreSQL (pgAdmin 4)** – Data storage and relational modeling  
- **Pentaho Data Integration** – ETL (Extraction, Transformation, Loading) process  
- **Tableau Desktop / Tableau Public** – OLAP analysis and data visualization  

### 🔄 ETL Process Overview
The ETL workflow automates the full data pipeline:
1. Extraction from the *Inside Airbnb* dataset (`listings.csv`)  
2. Data cleaning, type conversions, and feature engineering (e.g., price normalization, distance-to-coast calculation)
3. Loading into a **star schema** composed of:
   - `facts_table` (Airbnb listings)
   - `dim_hosts`
   - `dim_room_type_comb`
   - `dim_neighbourhood_group`

### 📊 Key Metrics
- **Dataset size:** ~22,752 listings (Lisbon, December 2023)  
- **ETL execution time:** ~23 seconds (17 seconds are related to API calls)  
- **Fully automated:** The ETL pipeline supports periodic updates and multi-city scalability  

### 📈 Interactive Dashboards
Two Tableau dashboards were created:

#### 1. Introduction Dashboard
- Average price per municipality  
- Top 10 hosts by number of listings  
- Distribution of listings by neighborhood and category  

#### 2. Detailed Dashboard
- Heatmaps of average prices by room type and location  
- Ratings distribution by neighborhood  
- Analysis of price vs. amenities (beds, bathrooms, capacity)  

---

## 🤖 Part 2 – Machine Learning

### 🎯 Objective
Extend the BI system with **predictive analytics** capabilities, using *data mining* to enhance pricing and occupancy insights.

### 🧠 Techniques Applied
#### 1. Price Recommendation (Classification Model)
- Predicts the optimal price category for a new listing based on its features  
- Algorithms tested: Decision Trees, Random Forests, KNN, AdaBoost  
- **Best model:** Decision Tree with ~80% accuracy for 10 price categories  

#### 2. Occupancy Forecasting (Time Series Analysis)
- Uses **ARIMA** to forecast accommodation demand and identify seasonal patterns  
- Example use cases: predicting high-demand periods (e.g., holidays, events)

### 🧰 Tools and Libraries
- **Python 3.11**
- **Pandas**, **NumPy**, **Scikit-learn**, **Statsmodels**, **Matplotlib**
- **Pentaho** (for data extraction)
- **PostgreSQL** (for DW integration)

### 📊 Key Results
- **Decision Tree Classifier**
  - Train Accuracy: `0.969`
  - Test Accuracy: `0.800`
  - F1-Score: `0.800`
- **ARIMA Forecasting**
  - Identified seasonality aligned with tourism peaks (e.g., Easter, concerts)

### 💻 End-User Interface
A simple **Graphical User Interface (GUI)** allows users to:
- Input listing characteristics  
- Obtain predicted price category  
- Understand how listing features affect pricing  

---

## 💡 Business Value
The developed system brings value to both **Airbnb** and its **hosts**.

- For **Airbnb**:  
  Personalized recommendation system that provides insights into the housing market.  
  Access to statistical data and the ability to anticipate brand-promoting events.  
  Better understanding of trends and seasonal demand changes.  

- For **Hosts**:  
  Access to valuable insights and recommendations to improve listing strategies.  
  Understand how property characteristics affect price.  
  Estimate potential earnings based on similar listings.  
  Adapt services to seasonal demand variations.  
  Compare personal performance with average market results. 

---

## 🚀 Future Work
- Expand to multiple cities (e.g., Porto, Madrid, Barcelona)  
- Incorporate external data (weather, events, proximity to landmarks)  
- Deploy as a **web-based BI + ML platform**

---

## 📚 References
- Dataset: [Inside Airbnb](http://insideairbnb.com/get-the-data/)  
- Tools: Tableau, Pentaho, PostgreSQL, Python (Scikit-learn, Statsmodels)  
- Project Guidelines: *University of Coimbra – Business Intelligence 2023/24*

---
