import itertools
import json
from configparser import ConfigParser
import psycopg2
import pydotplus
from geopy.geocoders import Nominatim
import numpy as np
import requests
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn import model_selection, linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier as KNN
import seaborn as sns
from sklearn import metrics
import plotly.offline as offline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from wordcloud import WordCloud
import string
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def load_config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return config


def connect(config):
    try:
        with psycopg2.connect(**config) as conn:
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def get_facts_table_pandas(conn, table_name):
    query = f"SELECT * FROM {table_name} order by id"

    df = pd.read_sql_query(query, conn)

    return df


def remove_outliers(conn, df):
    print("Size anted da remocao")
    print(df.shape)

    features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'numericprice', 'minimum_nights',
                'maximum_nights']

    removed_ids = []

    for i in range(len(features)):
        q1 = df[features[i]].quantile(0.25)
        q3 = df[features[i]].quantile(0.75)
        iqr = q3 - q1

        removed_indices = df[
            ~((df[features[i]] >= q1 - 1.5 * iqr) & (df[features[i]] <= q3 + 1.5 * iqr))].index
        removed_ids.extend(df.loc[removed_indices, 'id'])

    print("Número de linhas outliers removidas:")
    print(len(list(set(removed_ids))))

    remove_outliers_db(conn, list(set(removed_ids)))


def remove_outliers_db(conn, ids):
    cur = conn.cursor()
    
    query = f"DELETE FROM facts_table WHERE id IN ({', '.join(str(id) for id in ids)})"

    cur.execute(query)

    conn.commit()

    cur.close()


def get_aditional_information(latitude, longitude):
    
    geolocalizador = Nominatim(user_agent="bi_2024")

    local = geolocalizador.reverse((latitude, longitude))

    # Extraindo os detalhes
    place_rank = local.raw["place_rank"]
    importance = local.raw["importance"]

    return {
        "Place_rank": place_rank,
        "Importance": importance
    }


def get_POIs_nearby(latitude, longitude, raio=100):
    url = f"https://overpass-api.de/api/interpreter?data=[out:json];node(around:{raio},{latitude},{longitude})['amenity'];out;"
    response = requests.get(url)
    data = response.json()
    return data['elements']


def get_POIs_nearby_count(latitude, longitude, raio=100):
    url = f"https://overpass-api.de/api/interpreter?data=[out:json];node(around:{raio},{latitude},{longitude})['amenity'];out;"
    response = requests.get(url)
    data = response.json()
    if 'elements' in data:
        return len(data['elements'])
    return 0 


def get_info_location(latitude, longitude):
    # https://nominatim.org/release-docs/latest/customize/Ranking/
    # https://nominatim.org/release-docs/latest/customize/Importance/
    informacoes_local = get_aditional_information(latitude, longitude)

    for key, value in informacoes_local.items():
        print(f"{key}: {value}")

    print()

    # https://www.openstreetmap.org/
    num_pois = get_POIs_nearby_count(latitude, longitude)
    print(num_pois)


def get_quartiles(data):
    num_quartis = 4

    # Convert data to numpy array
    data_array = np.array(data)

    # Compute quartiles
    quartiles = np.percentile(data_array, np.linspace(0, 100, num=num_quartis + 1))

    print(np.linspace(0, 100, num=num_quartis + 1))
    print(quartiles)

    quartiles_rounded = np.round(quartiles[1:-1], 2)

    print(quartiles_rounded)

    return list(quartiles_rounded)


def get_categoria_preco(price, quartiles):
    res = []
    cats = [str(min(price)) + " - " + str(quartiles[0])]

    for i in range(1, len(quartiles)):
        cats.append(str(quartiles[i - 1]) + " - " + str(quartiles[i]))

    cats.append(str(quartiles[-1]) + " - " + str(max(price)))

    print(cats)
    print(quartiles)

    for p in price:
        t = True
        for i in range(len(quartiles)):
            if float(p) <= float(quartiles[i]):
                res.append(cats[i])
                t = False
                break
        if t:
            res.append(cats[-1])
    return res


def add_column(conn, table_name, column_name, data_type):
    try:
        # Open a cursor to perform database operations
        cur = conn.cursor()

        # Construct SQL query
        query = f"ALTER TABLE {table_name} ADD {column_name} {data_type};"

        # Execute SQL query
        cur.execute(query)
        conn.commit()

        # Close the cursor
        cur.close()

        print(f"Column '{column_name}' added to table '{table_name}'.")
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def update_price_cat_column(conn, table_name, cats, ids):
    cur = conn.cursor()

    # Construct SQL query
    for i, cat in enumerate(cats):
        query = f"UPDATE {table_name} SET price_cat = %s WHERE id = %s;"
        cur.execute(query, (cat, int(ids[i])))

    conn.commit()

    # Close the cursor
    cur.close()

    print(f"'price_cat' column updated in table '{table_name}'.")


def add_column_price_cat(conn, df):
    # Sorting by ID and by the "source_date" column
    df_sorted = df.sort_values(by=['id', 'source_date'])

    price = df_sorted['numericprice']
    ids = df_sorted['id']

    quartiles = get_quartiles(price)
    # print(quartiles)
    cat_price = get_categoria_preco(price, quartiles)
    # print(cat_price)

    add_column(conn, "facts_table", "price_cat", "Varchar (20)")
    update_price_cat_column(conn, 'facts_table', cat_price, ids)


# Best Parameters: {'criterion': 'entropy', 'max_depth': 34, 'min_samples_leaf': 1, 'min_samples_split': 2}
def analyzeDT_bp(df):
    features = ['latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'has_availability_bool',
                'instant_bookable_bool', 'accommodates', 'bathrooms', 'bedrooms', 'beds']
    
    # Select only the columns that can be converted to float
    X = df[features]

    # Applying one-hot encoding to the 'room_type' column
    room_type_encoded = pd.get_dummies(df['room_type'], prefix='room_type')

    # Concatenate the one-hot encoded room_type with X
    X = pd.concat([X, room_type_encoded], axis=1)

    y = df["price_cat"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    # Define the parameters grid
    param_grid = {
        'criterion': ['entropy'],
        'max_depth': list(range(20, 35, 2)),
        'min_samples_split': list(range(1, 10, 1)),
        'min_samples_leaf': list(range(1, 5, 1))
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Train the model with the best parameters
    dt = DecisionTreeClassifier(**best_params)
    dt.fit(X_train, y_train)

    y_pred_train = dt.predict(X_train)
    y_pred = dt.predict(X_test)

    print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))
    print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))


def analyzeDT(df):
    features = ['latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'has_availability_bool',
                'instant_bookable_bool', 'accommodates', 'bathrooms', 'bedrooms', 'beds']
    
    # Select only the columns that can be converted to float
    X = df[features]

    # Applying one-hot encoding to the 'room_type' column
    room_type_encoded = pd.get_dummies(df['room_type'], prefix='room_type')

    # Concatenate the one-hot encoded room_type with X
    X = pd.concat([X, room_type_encoded], axis=1)

    y = df["price_cat"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    dt = DecisionTreeClassifier(criterion='entropy', max_depth=34, min_samples_leaf=1, min_samples_split=2)
    dt.fit(X_train, y_train)

    y_pred_train = dt.predict(X_train)
    y_pred = dt.predict(X_test)

    print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))
    print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return dt


def analyzeKNN(df):
    scaller = StandardScaler()

    features = ['latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'has_availability_bool',
                'instant_bookable_bool', 'accommodates', 'bathrooms', 'bedrooms', 'beds']

    # Select only the columns that can be converted to float
    X = df[features]

    # Applying one-hot encoding to the 'room_type' column
    room_type_encoded = pd.get_dummies(df['room_type'], prefix='room_type')

    # Concatenate the one-hot encoded room_type with X
    X = pd.concat([X, room_type_encoded], axis=1)

    y = df["price_cat"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    X_train = scaller.fit_transform(X_train)
    X_test = scaller.fit_transform(X_test)

    knn = KNN(n_neighbors=1, weights='distance')
    knn.fit(X_train, y_train)

    y_pred_train = knn.predict(X_train)
    y_pred = knn.predict(X_test)

    print('Accuracy of KNN-Train: ', accuracy_score(y_pred_train, y_train))
    print('Accuracy of KNN-Test: ', accuracy_score(y_pred, y_test))

    ## -- Using cross-validation for parameter tuning

    # # creating list of K for KNN
    # k_list = list(range(1, 50, 2))
    # # creating list of cv scores
    # cv_scores = []
    #
    # # perform 10-fold cross validation
    # for k in k_list:
    #     knn = KNN(n_neighbors=k)
    #     scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    #     cv_scores.append(scores.mean())
    #
    # # changing to misclassification error
    # MSE = [1 - x for x in cv_scores]
    #
    # # finding best k
    # best_k = k_list[MSE.index(min(MSE))]
    # print("The optimal number of neighbors is %d." % best_k)


# Best hyperparameters: {'max_depth': 26, 'n_estimators': 81}
def analyzeForest(df):
    le = LabelEncoder()

    features = ['latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'has_availability_bool',
                'instant_bookable_bool', 'accommodates', 'bathrooms', 'bedrooms', 'beds']
    
    # Select only the columns that can be converted to float
    X = df[features]

    # Applying one-hot encoding to the 'room_type' column
    room_type_encoded = pd.get_dummies(df['room_type'], prefix='room_type')

    # Concatenate the one-hot encoded room_type with X
    X = pd.concat([X, room_type_encoded], axis=1)

    # Adjusting and transforming the class labels in the target dataset
    y_encoded = le.fit_transform(df["price_cat"])

    #Train the model
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_encoded, test_size=0.25, random_state=400,
                                                                        stratify=y_encoded)
    
    # param_grid = {
    #     'n_estimators': list(range(1, 100, 5)),
    #     'max_depth': list(range(1, 50, 5))
    # }

    # clf = RandomForestClassifier(random_state=200)
    #
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
    #
    # grid_search.fit(X_train, y_train)
    #
    # best_params = grid_search.best_params_
    # print("Melhores hiperparâmetros:", best_params)
    # 
    # # y_pred = grid_search.predict(X_test)
    # 
    # # print("Relatório de Classificação:")
    # # print(classification_report(y_test, y_pred))

    reg = RandomForestClassifier(n_estimators=81, max_depth=26, oob_score=True, random_state=200)
    reg.fit(X_train, y_train)

    y_pred_train = reg.predict(X_train)
    y_pred = reg.predict(X_test)

    print('Accuracy of RandomForests-Train: ', accuracy_score(y_pred_train, y_train))
    print('Accuracy of RandomForests-Test: ', accuracy_score(y_pred, y_test))

    # # Get feature importances
    # importances = reg.feature_importances_
    #
    # # Create a DataFrame to store the importances along with the feature names
    # feature_importance_df = pd.DataFrame(list(zip(X.columns, importances)), columns=['Feature', 'Importance'])
    #
    # # Sort the features by importance in descending order
    # feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    #
    # # Display the ordered list of features
    # print(feature_importance_df)
    #
    # print(classification_report(y_test, y_pred))
    #
    # # print(confusion_matrix(y_train, y_pred_train))
    # print(confusion_matrix(y_test, y_pred))

    return reg


# Best parameters n_estimators 50, learning_rate: 0.6
def analyseAdaBoost(df):

    le = LabelEncoder()

    y_encoded = le.fit_transform(df["price_cat"])

    features = ['latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'has_availability_bool',
                'instant_bookable_bool', 'accommodates', 'bathrooms', 'bedrooms', 'beds']

    X = df[features]

    room_type_encoded = pd.get_dummies(df['room_type'], prefix='room_type')

    X = pd.concat([X, room_type_encoded], axis=1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_encoded, test_size=0.25, random_state=400)

    reg = AdaBoostClassifier(n_estimators=50, random_state=200, learning_rate=0.6)
    reg.fit(X_train, y_train)

    y_pred_train = reg.predict(X_train)
    y_pred = reg.predict(X_test)

    print('Accuracy of AdaBoost-Train: ', accuracy_score(y_pred_train, y_train))
    print('Accuracy of AdaBoost-Test: ', accuracy_score(y_pred, y_test))

    # param_grid = {
    #     'n_estimators': np.arange(0, 100, 5),
    #     'learning_rate': np.arange(0.1, 1.1, 0.1)  # Valores entre 0 e 1 com incrementos de 0.1
    # }
    #
    # clf = AdaBoostClassifier(random_state=200)
    #
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
    #
    # grid_search.fit(X_train, y_train)
    #
    # best_params = grid_search.best_params_
    # print("Melhores hiperparâmetros:", best_params)
    #
    # y_pred = grid_search.predict(X_test)
    #
    # print("Relatório de Classificação:")
    # print(classification_report(y_test, y_pred))


def analyseLinearRegression(df):
    features = ['latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'has_availability_bool',
                'instant_bookable_bool', 'accommodates', 'bathrooms', 'bedrooms', 'beds']

    X = df[features]

    room_type_encoded = pd.get_dummies(df['room_type'], prefix='room_type')

    X = pd.concat([X, room_type_encoded], axis=1)

    y = df["numericprice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)


    train_accuracy = reg.score(X_train, y_train)
    test_accuracy = reg.score(X_test, y_test)
    print('Accuracy of LinearRegression-Train: ', train_accuracy)
    print('Accuracy of LinearRegression-Test: ', test_accuracy)

    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Mean Squared Error:', mse)
    print('R-squared:', r2)


def show_correlation_matrix(df):
    # Selecting 'numericprice' and other columns
    selected_columns = ['numericprice'] + [col for col in df.columns if col != 'numericprice']
    selected_df = df[selected_columns]

    # Calculate the correlation matrix
    correlation_matrix = selected_df.corr(numeric_only=True)

    # Sort correlation matrix
    sorted_correlation = correlation_matrix['numericprice'].sort_values(ascending=False)

    # Show correlation matrix
    print(sorted_correlation)
    


def predict_price(dt, params, entry_vars, room_type_var):
    dic = {}
    for item in params:
        param = entry_vars[item].get()
        dic[item] = param

    room_t = room_type_var.get()
    room_types = ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room']
    for i, room_type in enumerate(room_types):
        dic['room_type_' + room_type] = 1 if room_t == i else 0

    df1 = pd.DataFrame.from_dict(dic, orient='index').T
    predicted_price = dt.predict(df1)
    messagebox.showinfo("Prediction", 'Predicted Price Category: ' + str(predicted_price[0]) + ' $')


def create_GUI(dt):
    # Create GUI
    root = tk.Tk()
    root.title("Price Recommendation System")

    # Load and resize image
    original_image = Image.open("Airbnb_Logo.png")
    width, height = 300, 94
    resized_image = original_image.resize((width, height))
    photo = ImageTk.PhotoImage(resized_image)
    label = tk.Label(root, image=photo)
    label.pack()

    params = ['latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'has_availability_bool',
              'instant_bookable_bool', 'accommodates', 'bathrooms', 'bedrooms', 'beds']

    entry_vars = {}
    for item in params:
        frame = tk.Frame(root)
        frame.pack()
        label = tk.Label(frame, text=item)
        label.pack(side=tk.LEFT)
        entry_var = tk.StringVar()
        entry = tk.Entry(frame, textvariable=entry_var)
        entry.pack(side=tk.LEFT)
        entry_vars[item] = entry_var

    room_type_var = tk.IntVar()
    room_type_label = tk.Label(root, text="Room Type:")
    room_type_label.pack()
    for i, room_type in enumerate(['Entire home/apt', 'Hotel room', 'Private room', 'Shared room']):
        tk.Radiobutton(root, text=room_type, variable=room_type_var, value=i).pack()

    predict_button = tk.Button(root, text="Predict Price",
                               command=lambda: predict_price(dt, params, entry_vars, room_type_var))
    predict_button.pack()

    root.mainloop()


def time_series(cidade):
    listing_calendar = pd.read_csv("calendar_" + cidade + "_18mar2024.csv")
    reviews_s = pd.read_csv("reviews_" + cidade + "_18mar2024.csv")

    # Group by listing ID and count unique price values
    listings_price_counts = listing_calendar.groupby('listing_id')['price'].nunique()

    # Count the number of listings with only one unique price value
    constant_price_listings = (listings_price_counts > 1).sum()

    print(constant_price_listings)

    # # https://github.com/xavierfactor/Airbnb-TimeSeries-Forecasting/blob/main/Airbnb_Supply_and_Demand_Analytics.ipynb

    # # Convert dates into datetime format
    listing_calendar['date'] = pd.to_datetime(listing_calendar['date'])

    # Number of Occupied Listings Over Time:

    occupied_listings_over_time = listing_calendar[listing_calendar['available'] == 'f'].groupby('date')[
        'listing_id'].count()

    # print(occupied_listings_over_time)

    plt.figure(figsize=(10, 6))
    occupied_listings_over_time.plot(kind='line')
    plt.title('Number of Listings Occupied Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Occupied Listings')
    plt.show()

    ## Only to may 2024

    may_data = listing_calendar[(listing_calendar['date'].dt.month == 5) &
                                (listing_calendar['date'].dt.year == 2024) &
                                (listing_calendar['date'].dt.day >= 15)]

    occupied_listings_over_time = may_data[may_data['available'] == 'f'].groupby('date')['listing_id'].count()

    plt.figure(figsize=(10, 6))
    occupied_listings_over_time.plot(kind='line')
    plt.title('Number of Listings Occupied in May 2024')
    plt.ylabel('Number of Occupied Listings')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

    # Number of Listings Over Time (by Day of the Week):

    active_listings_over_time_day = listing_calendar[listing_calendar['available'] == 'f'].groupby(
        listing_calendar['date'].dt.dayofweek)['listing_id'].count()

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    active_listings_over_time_day.index = [day_names[index] for index in active_listings_over_time_day.index]

    plt.figure(figsize=(10, 6))
    active_listings_over_time_day.plot(kind='line')
    plt.title('Number of Occupied Listings Over Time (days of the week)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Occupied Listings')
    plt.tight_layout()
    plt.show()

    ## -- Market Demand or Estimated Occupancy

    reviews_s['date'] = pd.to_datetime(reviews_s['date'])

    reviews_s['month'] = reviews_s['date'].dt.month

    reviews_per_month = reviews_s.groupby('month').size().reset_index(name='review_count')

    month_names = pd.date_range(start='2024-01-01', periods=12, freq='MS').strftime('%B').tolist()

    reviews_per_month['month'] = reviews_per_month['month'].map(dict(zip(range(1, 13), month_names)))

    plt.figure(figsize=(10, 6))
    plt.bar(reviews_per_month['month'], reviews_per_month['review_count'])
    plt.title('Number of reviews per month')
    plt.xlabel('Month')
    plt.ylabel('Number of reviews')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Average monthly review per listing
    # Convert the 'date' column in the reviews dataframe to datetime
    reviews_s['date'] = pd.to_datetime(reviews_s['date'])

    # Extract month and year from the 'date' column and create a new 'year_month' column
    reviews_s['year_month'] = reviews_s['date'].dt.to_period('M')

    # Group by 'listing_id' and 'year_month', count the number of reviews, then reset index
    reviews_per_month = reviews_s.groupby(['listing_id', 'year_month']).size().reset_index(name='review_count')

    # Group by 'year_month' and calculate the average number of reviews per listing
    avg_reviews_per_month = reviews_per_month.groupby('year_month')['review_count'].mean().reset_index()

    # print(avg_reviews_per_month)

    # Convert 'year_month' to datetime for plotting
    avg_reviews_per_month['year_month'] = avg_reviews_per_month['year_month'].astype('datetime64[ns]')

    # To select data from 2016 onwards
    # avg_reviews_per_month = avg_reviews_per_month.loc[(avg_reviews_per_month['year_month'].dt.year >= 2016)]

    plt.figure(figsize=(10, 6))
    plt.plot(avg_reviews_per_month['year_month'], avg_reviews_per_month['review_count'])
    plt.title('Average Monthly Reviews per Listing')
    plt.xlabel('Year')
    plt.ylabel('Average Reviews')
    plt.grid()
    plt.show()

    ## ------ ARIMA

    # Split training and test data (2016 to 2019 for training)
    train_data = avg_reviews_per_month.loc[
        (avg_reviews_per_month['year_month'].dt.year >= 2016) & (avg_reviews_per_month['year_month'].dt.year < 2020)]
    test_data = avg_reviews_per_month.loc[
        (avg_reviews_per_month['year_month'].dt.year >= 2022) & (avg_reviews_per_month['year_month'].dt.year < 2024)]
    covid = avg_reviews_per_month.loc[
        (avg_reviews_per_month['year_month'].dt.year == 2020) | (avg_reviews_per_month['year_month'].dt.year == 2021)]

    # Use only the 'review_count' column as the endogenous variable
    train_data = train_data.set_index('year_month')['review_count']
    test_data = test_data.set_index('year_month')['review_count']
    covid = covid.set_index('year_month')['review_count']

    # Modelo ARIMA
    model = ARIMA(train_data, order=(9, 2, 6))
    model_fit = model.fit()

    # Best ARIMA parameters: (9, 0, 7) -  MSE (LISBON)
    # Best ARIMA parameters: (9, 2, 6) -  MSE (PORTO)
    # # Define a range of values for p, d, and q
    # p_values = range(0, 10)  # Order of the autoregressive (AR) component
    # d_values = range(0, 10)  # Degree of differencing
    # q_values = range(0, 10)  # Order of the moving average (MA) component
    #
    # # Create all possible combinations of p, d, and q
    # pdq_combinations = list(itertools.product(p_values, d_values, q_values))
    #
    # best_aic = float('inf')
    # best_params = None
    #
    # # Perform a grid search to find the best parameters
    # for pdq in pdq_combinations:
    #     try:
    #         model = ARIMA(train_data, order=pdq)
    #         model_fit = model.fit()
    #         # aic = model_fit.aic
    #
    #         predictions = model_fit.forecast(steps=60)  # 24 months in 2024
    #
    #         # Evaluate the model
    #         mse = mean_squared_error(test_data, predictions[24:-12])  # Calculate the MSE only for 2023 data
    #
    #         # Update the best parameters if the current AIC is better
    #         if mse < best_aic:
    #             best_aic = mse
    #             best_params = pdq
    #     except:
    #         continue
    #
    # print("Best ARIMA parameters:", best_params)
    # print("Best AIC:", best_aic)

    # Make predictions
    predictions = model_fit.forecast(steps=60)

    # Evaluate the model
    mse = mean_squared_error(test_data, predictions[24:-12])
    print('Mean Squared Error:', mse)

    print(predictions[-12:])

    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Train')
    plt.plot(test_data.index, test_data, label='Test')
    plt.plot(covid.index, covid, label='Covid', color='green')
    plt.plot(predictions.index[24:], predictions[24:], label='Predictions', color='red')
    plt.title('ARIMA Model - Average Monthly Reviews per Listing')
    plt.xlabel('Year')
    plt.ylabel('Average Reviews')
    plt.legend()
    plt.grid()
    plt.show()

    ## ----- END OF ARIMA

    ## https://ourworldindata.org/grapher/average-length-of-stay?tab=chart&time=1995..latest&country=~PRT
    avg_length_of_stay = 3.2
    review_rate_multiplier = 1
    avg_reviews_per_month['occupancy_sf'] = (avg_reviews_per_month[
                                                 'review_count'] * review_rate_multiplier * avg_length_of_stay) / 30

    predictions = (predictions * review_rate_multiplier * avg_length_of_stay) / 30

    plt.figure(figsize=(10, 6))
    plt.plot(avg_reviews_per_month['year_month'], avg_reviews_per_month['occupancy_sf'], label='From real data')
    plt.plot(predictions.index[-12:], predictions[-12:], label='Preditions', color='red')
    plt.title('Prediction of Average Occupancy Rate')
    plt.xlabel('Year')
    plt.ylabel('Occupancy Rate')
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate the mean for 2022

    # Filter rows for the year 2022
    avg_reviews_2022 = avg_reviews_per_month[avg_reviews_per_month['year_month'].dt.year == 2022]

    # Calculate occupancy_sf for 2022
    avg_reviews_2022 = ((avg_reviews_2022[
                             'review_count'] * review_rate_multiplier * avg_length_of_stay) / 30).mean()

    print("avg_reviews_2022")
    print(avg_reviews_2022)


def time_series_2(df):
    # Price per month
    # Convert the 'last_review' column to datetime
    df['source_date'] = pd.to_datetime(df['source_date'])

    # Group by 'month' and calculate the average price
    price_over_time_cv = df.groupby('source_date')['numericprice'].mean()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(price_over_time_cv.index, price_over_time_cv.values, marker='o', linestyle='-')
    plt.title('Average price across different files')
    plt.xlabel('Month')
    plt.ylabel('Average price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    config = load_config()
    con = connect(config)
    df = get_facts_table_pandas(con, 'facts_table')

    # obter_info_loc("38.708", "-9.171")
    #
    # remove_outliers(con, df)
    #
    # add_column_price_cat(con, df)
    #
    # time_series('lisboa')
    #
    # time_series_2(df)
    #
    # show_correlation_matrix(df)

    dt = analyzeDT(df)

    # analyzeKNN(df)
    # analyzeForest(df)
    # analyseAdaBoost(df)
    # analyseLinearRegression(df)

    create_GUI(dt)

    con.close()
