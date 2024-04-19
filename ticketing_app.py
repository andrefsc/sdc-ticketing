from sklearn.externals import joblib
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
import os

# Igonre UserWarnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set the current path and load the saved pipeline
current_directory = os.path.dirname(os.path.abspath(__file__))

def get_base64_of_image(image_path):
    """
    Encode the image at the specified path to Base64 format.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The Base64-encoded representation of the image.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def add_bg_image_with_transparency(image_base64, opacity=0.5):  
    """
    Add a background image with transparency to the Streamlit app.

    Args:
        image_base64 (str): The Base64-encoded representation of the background image.
        opacity (float): The opacity level of the overlay (between 0 and 1). Default is 0.5.

    Returns:
        None
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, {opacity});  # White with opacity for transparency effect
            pointer-events: none;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

image_path = os.path.join(current_directory, 'image01.jpg') 
image_base64 = get_base64_of_image(image_path)
add_bg_image_with_transparency(image_base64, opacity=0.1)  
image_path = os.path.join(current_directory, 'image02.png')  
st.image(image_path, width=200, use_column_width=False) 

# Load the model and the OneHotEncoder
model_and_encoder = joblib.load('zP_model_and_encoder.pkl')
model_and_encoder1 = joblib.load('zA_model_and_encoder.pkl')
# Extract the model and the encoder from the dictionary
model = model_and_encoder['model']
encoder = model_and_encoder['encoder']
model1 = model_and_encoder1['model']
encoder1 = model_and_encoder1['encoder']

# Opções para a caixa de seleção (Dropbox)
competition_types =  ['National League', 'Nation Cup/Play-off', 'Continental Cup' ,'Friendly']
hour_games = ['afternoon', 'evening']
weekday_types = ['weekdaygame', 'weekendgame']
game_classes = ['Classic', 'Regular']

# Streamlit app
st.title('FanFair: Precios Justos a Través de ML para Entradas de Estadios')

# Seleção do usuário
selected_competition_type = st.selectbox('Tipo de Competencia', competition_types)
selected_hour_game = st.selectbox('Hora del Juego', hour_games)
selected_weekday_type = st.selectbox('Tipo de Día de la Semana', weekday_types)
selected_game_class = st.selectbox('Clase del Juego', game_classes)

# Atualiza base_data com as seleções do usuário
base_data = {
    'Competition_type': selected_competition_type,
    'hour_game': selected_hour_game,
    'weekday_type': selected_weekday_type,
    'game_class': selected_game_class,
}

# Botão para realizar predições
if st.button('Generar predicción de Precios'):
    # Price categories for prediction
    price_categories = ['Low price', 'Medium price', 'High price', 'Premium']

    # Function to prepare the input data
    def prepare_data(input_data, encoder, price_category):
        input_data_full = input_data.copy()
        input_data_full['price_category'] = price_category
        data_to_encode = np.array(list(input_data_full.values())).reshape(1, -1)
        encoded_data = encoder.transform(data_to_encode).toarray()
        return encoded_data

    predictions = []

    # Loop to collect predictions for each price category
    for category in price_categories:
        prepared_data = prepare_data(base_data, encoder, category)
        prediction = model.predict(prepared_data)
        predictions.append((category, prediction[0]))

    # Sort predictions by predicted price in ascending order
    sorted_predictions = sorted(predictions, key=lambda x: x[1])

    attendance_predictions = []

    # Defining a dictionary with percentage ranges for minimum and maximum by price category
    percentage_range = {
        'Low price': {'min': 0.10, 'max': 0.12},  # Example: 10% for minimum, 12% for maximum in 'Low price' category
        'Medium price': {'min': 0.08, 'max': 0.10},  # 8% for minimum, 10% for maximum in 'Medium price'
        'High price': {'min': 0.05, 'max': 0.07},  # 5% for minimum, 7% for maximum in 'High price'
        'Premium': {'min': 0.03, 'max': 0.05}  # 3% for minimum, 5% for maximum in 'Premium'
    }

    # Revenue summaries
    revenue_summary = {
        'Predicted Revenue': 0,
        'Min Revenue': 0,
        'Max Revenue': 0
    }

    # assistance
    assistance_counter = {}

    for category, price in sorted_predictions:
        st.write("\n")
        rounded_prediction = round(price, 2)
        min_percentage = percentage_range[category]['min']
        max_percentage = percentage_range[category]['max']
        min_prediction = round(rounded_prediction - (rounded_prediction * min_percentage), 2)
        max_prediction = round(rounded_prediction + (rounded_prediction * max_percentage), 2)

        # Preparation for attendance prediction
        categorical_data_to_encode = [[
            base_data['Competition_type'],
            base_data['hour_game'],
            base_data['weekday_type'],
            base_data['game_class'],
            category,
        ]]
        
        encoded_categorical_data = encoder1.transform(categorical_data_to_encode).toarray()
        encoded_data = np.hstack((encoded_categorical_data, [[price]]))
        attendance_prediction = model1.predict(encoded_data)
        attendance_predictions.append(attendance_prediction[0])
        
        # Round the attendance prediction to the nearest whole number before adding it to the list
        rounded_attendance_prediction = round(attendance_prediction[0])
        attendance_predictions.append(rounded_attendance_prediction)
        
        st.markdown(f"**Categoria de Précio: {category}**")

        # Printing the combined results with the rounded attendance prediction
        st.write(f"Precio unitario del ticket para {category}: **{rounded_prediction}** (mín-máx: {min_prediction} - {max_prediction})")
        st.write(f"Asistencia predicha: **{rounded_attendance_prediction}** aficionados")


        st.write("*")
    
        # Calculate revenue for predicted, minimum, and maximum prices
        revenue_predicted = round(rounded_prediction * rounded_attendance_prediction,2)
        revenue_min = round(min_prediction * rounded_attendance_prediction,2)
        revenue_max = round(max_prediction * rounded_attendance_prediction,2)
        
        # Printing the combined results with the revenue calculation
        st.write(f"Precio del ticket predicho: {rounded_prediction}  |  Ingresos esperados: {revenue_predicted}")
        st.write(f"Precio del ticket mínimo: {min_prediction}  |  Ingresos esperados: {revenue_min}")
        st.write(f"Precio del ticket máximo: {max_prediction}  |  Ingresos esperados: {revenue_max}\n")

        st.markdown('<hr style="border:1px solid #021a40">', unsafe_allow_html=True)

        # Update revenue summaries
        revenue_summary['Predicted Revenue'] += round(revenue_predicted,2)
        revenue_summary['Min Revenue'] += round(revenue_min,2)
        revenue_summary['Max Revenue'] += round(revenue_max,2)

    # Displaying the revenue summary table
    df_revenue_summary = pd.DataFrame([revenue_summary])
    #st.write(df_revenue_summary.to_string(index=False))
    #st.write("\n")

    # Supondo que revenue_summary já foi definido anteriormente no seu código
    for key, value in revenue_summary.items():
        # Formatar o valor para incluir o símbolo de dólar e duas casas decimais
        # Aqui, estamos assumindo que os valores já estão em formato de número float
        #formatted_value = f"${value:,.2f}".replace(",", " ")
        st.write(f"{key}: {value}")

 
    # Calculating the sum of all elements in the vector
    sum_of_attendance_predictions = int(round(sum(attendance_predictions),0))
    st.markdown('<hr style="border:1px solid #021a40">', unsafe_allow_html=True)
    # Printing the sum of the vector
    st.markdown(f"La asistencia total esperada es de **{sum_of_attendance_predictions}** aficionados.")

