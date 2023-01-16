import streamlit as st
import plotly.express as px
from backend import get_data
import base64
import plotly.io as pio

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
            Pressure = [dict["main"]["humidity"] for dict in filtered_data]
            dates = [dict["dt_txt"] for dict in filtered_data]
            figure = px.line(x=dates, y=Pressure, labels={"x": "Date", "y": "Humidity"})
            st.plotly_chart(figure)

        if option == "Visibility":
            Visibility = [dict["main"]["humidity"] for dict in filtered_data]
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


