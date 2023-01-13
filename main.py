import streamlit as st
import plotly.express as px
from backend import get_data


import base64


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

st.title("Weather Forecast for the next days")

place = st.text_input("place: ")

days = st.slider("Forecast Days", min_value=1, max_value=5, help="select the number of forecasted days")

option = st.selectbox("select data to view", ("Temperature", "Sky"))

st.subheader(f"{option} for the next {days} days in {place}")

data = get_data(place, forecast_days, kind)
d, t = get_data(days)

figure = px.line(x=d, y=t, labels={"x": "Date", "y": "Temperature (C)"})
st.plotly_chart(figure)
