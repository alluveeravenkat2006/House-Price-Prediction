import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Hyderabad House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

page_bg = """
<style>

[data-testid="stAppViewContainer"] {
    background-image: url("https://static.vecteezy.com/system/resources/previews/037/354/231/large_2x/ai-generated-luxury-modern-home-design-ideas-black-exterior-home-exterior-free-photo.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.main-box {
    background: rgba(0,0,0,0.65);
    padding: 30px;
    border-radius: 20px;
    color: white;
    box-shadow: 0px 0px 25px rgba(0,0,0,0.5);
}

.title {
    text-align: center;
    color: #ffffff;
    font-size: 45px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #f1f1f1;
    font-size: 20px;
}

.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 20px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
}

.prediction-box {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

data = pd.read_csv("Hyderabad_House_Data.csv")

for col in data.columns:

    if data[col].dtype == 'object':

        data[col] = data[col].astype(str).str.strip()
        
data['Bedrooms'] = (
    data['Bedrooms']
    .astype(str)
    .str.extract(r'(\d+)')[0]
)

data['Bedrooms'] = pd.to_numeric(
    data['Bedrooms'],
    errors='coerce'
)

data['Washrooms'] = pd.to_numeric(
    data['Washrooms'],
    errors='coerce'
)

data['Area'] = (
    data['Area']
    .astype(str)
    .str.replace('sqft', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)

data['Area'] = pd.to_numeric(
    data['Area'],
    errors='coerce'
)

data['Price'] = (
    data['Price']
    .astype(str)
    .str.replace(',', '', regex=False)
)

data['Price'] = pd.to_numeric(
    data['Price'],
    errors='coerce'
)

data = data.fillna(0)

label_encoders = {}

object_cols = data.select_dtypes(include='object').columns

for col in object_cols:

    le = LabelEncoder()

    data[col] = le.fit_transform(
        data[col].astype(str)
    )

    label_encoders[col] = le

X = data.drop('Price', axis=1)

y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = r2_score(y_test, y_pred)

st.markdown(
    '<div class="main-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="title">🏠 Hyderabad House Price Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict house prices using Machine Learning & Random Forest Algorithm</p>',
    unsafe_allow_html=True
)

st.write("---")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="Model Accuracy (R2 Score)",
        value=round(accuracy, 2)
    )

with col2:

    st.metric(
        label="Dataset Rows",
        value=len(data)
    )

st.write("---")

left_col, right_col = st.columns(2)

with left_col:

    bedrooms = st.slider(
        "Bedrooms",
        1,
        10,
        2
    )

    washrooms = st.slider(
        "Washrooms",
        1,
        10,
        2
    )

    area = st.number_input(
        "Area (sqft)",
        min_value=100,
        max_value=10000,
        value=1200
    )

with right_col:

    furnishing = st.selectbox(
        "Furnishing",
        label_encoders['Furnishing'].classes_
    )

    tenant = st.selectbox(
        "Tenant Type",
        label_encoders['Tennants'].classes_
    )

    locality = st.selectbox(
        "Locality",
        label_encoders['Locality'].classes_
    )

furnishing_encoded = label_encoders[
    'Furnishing'
].transform([furnishing])[0]

tenant_encoded = label_encoders[
    'Tennants'
].transform([tenant])[0]

locality_encoded = label_encoders[
    'Locality'
].transform([locality])[0]


if st.button("Predict House Price"):
    input_data = pd.DataFrame({
        'Bedrooms': [bedrooms],
        'Washrooms': [washrooms],
        'Area': [area],
        'Furnishing': [furnishing_encoded],
        'Tennants': [tenant_encoded],
        'Locality': [locality_encoded]
    })

    prediction = model.predict(input_data)

    st.markdown(
        f'''<div class="prediction-box">
        Predicted House Price<br><br>
        ₹ {round(prediction[0], 2)}
        </div>''',
        unsafe_allow_html=True
    )

st.write("---")

st.success("Project Developed using Streamlit + Machine Learning")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    '''
    <div class="footer">
        Built with Machine Learning • Powered by Streamlit • 
        Designed by 
        <span class="highlight-name">
            A.v.v.satyanarayana
        </span>
    </div>
    ''',
    unsafe_allow_html=True
)