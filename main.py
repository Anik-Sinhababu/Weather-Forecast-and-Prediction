import streamlit as st
import plotly.express as px
from backend import get_data
import base64
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import re
import warnings

warnings.filterwarnings('ignore')
import missingno as mso
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


# to set image at background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


add_bg_from_local('img_1.png')

# add the user interface
menu = st.sidebar.radio("Menu", ["Weather Forecast", "Weather Prediction And Visualization"])
if menu == "Weather Forecast":
    st.title("Weather Forecast for the next days")

    place = st.text_input("place: ")

    days = st.slider("Forecast Days", min_value=1, max_value=5, help="select the number of forecasted days")

    option = st.selectbox("select data to view", ("Temperature", "Sky", "Humidity", "Pressure", "Visibility"))

    st.subheader(f"{option} for the next {days} days in {place}")


    # create a temperature plot
    if place:
        # get weather data temperature/sky data
        try:
            filtered_data = get_data(place, days)

            if option == "Temperature":
                temperatures = [dict["main"]["temp"] for dict in filtered_data]
                for i in range(len(temperatures)):
                    temperatures[i] -= 273

                dates = [dict["dt_txt"] for dict in filtered_data]
                figure = px.line(x=dates, y=temperatures, labels={"x": "Date", "y": "Temperature (C)"})
                st.plotly_chart(figure)


            if option == "Humidity":
                Humidity = [dict["main"]["humidity"] for dict in filtered_data ]
                dates = [dict["dt_txt"] for dict in filtered_data]
                figure = px.line(x=dates, y=Humidity, labels={"x": "Date", "y": "Humidity"})
                pio.templates.default = "simple_white"

                px.defaults.template = "ggplot2"
                px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
                px.defaults.width = 1000
                px.defaults.height = 500

                df = px.data.iris()
                figure.show()

            if option == "Pressure":
                Pressure = [dict["main"]["pressure"] for dict in filtered_data]
                dates = [dict["dt_txt"] for dict in filtered_data]
                figure = px.line(x=dates, y=Pressure, labels={"x": "Date", "y": "Humidity"})
                st.plotly_chart(figure)

            if option == "Visibility":
                Visibility = [dict["main"]["visibility"] for dict in filtered_data]
                dates = [dict["dt_txt"] for dict in filtered_data]
                figure = px.line(x=dates, y=Visibility, labels={"x": "Date", "y": "Visibility"})
                st.plotly_chart(figure)

            if option == "Sky":
                def add_bg_from_local(image_file):
                    with open(image_file, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read())
                    st.markdown(
                        f"""
                    <style>
                    .stApp {{
                        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                        background-size: cover
                    }}
                    </style>
                    """,
                        unsafe_allow_html=True
                    )

                add_bg_from_local('img.png')
                images = {"Clear": "images_3/clear.png", "Clouds": "images_3/cloud.png", "Rain": "images_3/rain.png", "Snow":
                          "images_3/snow.png"}
                dates = [dict["dt_txt"] for dict in filtered_data]
                sky_conditions = [dict["weather"][0]["main"] for dict in filtered_data]

                image_paths = [images[condition] for condition in sky_conditions]
                st.image(image_paths, dates, width=150)
        except KeyError:
            st.info("This place does not exists, in the list try entering another location", icon="⚠️")

if menu == "Weather Prediction And Visualization":
    add_bg_from_local("R.jpg")
    data = pd.read_csv("seattle-weather.csv")
    st.header("Tabular Data")
    if st.checkbox("Tabular Data"):
        q = st.slider("Select the number of data you want to see", min_value=5, max_value=50)
        st.table(data.head(q))
    st.header("Statistics Data")
    if st.checkbox("Statistics"):
        st.table(data.describe())
    st.header("Correlation Graph")
    fig3 = plt.figure(figsize=(12, 6))
    sns.heatmap(data.corr(), cmap='coolwarm')
    st.pyplot(fig3)
    st.header("Graphs")
    graph = st.selectbox("Different Types Of Graphs",["Count Plot", "Scatter Plot", "Histogram Plot", "Pearson's Correlation"])
    if graph == "Count Plot":
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x='weather', data=data)
        st.pyplot(fig)
        sns.set(style='darkgrid')
    if graph == "Histogram Plot":
        fig1, axs = plt.subplots(2, 2, figsize=(10, 8))
        sns.histplot(data=data, x='precipitation', kde=True, ax=axs[0, 0], color='green')
        sns.histplot(data=data, x='temp_max', kde=True, ax=axs[0, 1], color='red')
        sns.histplot(data=data, x='temp_min', kde=True, ax=axs[1, 0], color='blue')
        sns.histplot(data=data, x='wind', kde=True, ax=axs[1, 1], color='orange')
        st.pyplot(fig1)
    if graph == "Scatter Plot":
        fig4, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=data, x="date", y="temp_max", hue="wind")
        st.pyplot(fig4)
    if graph == "Pearson's Correlation":
        data.plot("precipitation", 'temp_max', style='o')
        st.write('pearsons correlation: ', data['precipitation'].corr(data['temp_max']))
        st.write('T test and P value: ', stats.ttest_ind(data['precipitation'], data['temp_max']))
    countrain = len(data[data.weather == 'rain'])
    countsun = len(data[data.weather == 'sun'])
    countdrizzle = len(data[data.weather == 'drizzle'])
    countsnow = len(data[data.weather == 'snow'])
    countfog = len(data[data.weather == 'fog'])
    data = data.drop(['date'], axis=1)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    import numpy as np
    data.precipitation = np.sqrt(data.precipitation)
    data.wind = np.sqrt(data.wind)
    sns.set(style='darkgrid')
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    lc = LabelEncoder()
    data['weather'] = lc.fit_transform(data['weather'])
    x = ((data.loc[:, data.columns != 'weather']).astype(int)).values[:, 0:]
    y = data['weather'].values
    data.weather.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)
    p = st.number_input("Enter Precipitation", 0.00, 4.50, step=0.20)
    tmax = st.number_input("Enter Max Temperature", 0.4, 30.6, step=0.40)
    tmin = st.number_input("Enter Min Temperature", -4.0, 12.8, step=0.40)
    w = st.number_input("Enter Wind", 1.140, 4.000, step=0.20)
    ot = xgb.predict([[p, tmax, tmin, w]])
    print('the weather is:')
    if st.button("Predict"):
        if ot == 0:
            st.write('Today it will Drizzle')
            st.subheader("images_3/cloud.png", width=125)
        elif ot == 1:
            st.subheader('Its foggy today')
        elif ot == 2:
            st.subheader('It will rain')
            st.image("images_3/rain.png", width=125)
        elif ot == 3:
            st.subheader('today it will snow')
            st.image("images_3/snow.png", width=125)
        else:
            st.subheader('sunny')
            st.image("images_3/clear.png", width=125)

    import pickle
    file = 'model.pkl'
    pickle.dump(xgb, open(file, 'wb'))
