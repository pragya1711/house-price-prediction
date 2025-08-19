import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
from openai import OpenAI
import os

# Initialize Gemini (OpenAI-compatible) model
client = OpenAI(
    api_key="AIzaSyB2nLtRLzxqRDXAUfwwsRiBokiiYOsBnEU",  # Replace with your API key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Load your CSV file
data = pd.read_csv("real_estate_prices_updated1.csv")

# Features (X) and Target (y)
X = data[["city", "location", "area_sqft", "bedrooms", "furnishing", "age_years"]]
y = data["price_in_lakhs"]
    
# Convert categorical to numeric (one-hot encoding)
X = pd.get_dummies(X, dtype=int)

# Train model
model = LinearRegression()
model.fit(X, y)


# ---- Prediction Function ----
def predict_price(city, location, area_sqft, bedrooms, furnishing, age_years):
    try:
        # Input validation
        if area_sqft < 400 or area_sqft > 10000:
            return "⚠ Area outside normal range."
        if bedrooms > area_sqft / 100:
            return "⚠ Too many bedrooms for given area."
        if age_years < 0 or age_years > 100:
            return "⚠ Age of property is unrealistic."

        # Standardize text
        city = city.strip().title()
        location = location.strip().title()
        furnishing = furnishing.strip().title()

        # Create dataframe
        input_data = pd.DataFrame(
            [[city, location, area_sqft, bedrooms, furnishing, age_years]],
            columns=["city", "location", "area_sqft", "bedrooms", "furnishing", "age_years"]
        )

        # One-hot encode
        input_data = pd.get_dummies(input_data, dtype=int)

        # Align with training columns
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[X.columns]

        # Predict
        prediction = float(model.predict(input_data)[0])

        # Apply minimum floor
        min_price = float(data["price_in_lakhs"].min())
        if prediction < min_price:
            prediction = min_price

        return f"Estimated Price: {prediction:.2f} lakhs"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ---- AI Chat Function ----
def price_pandit(myquestion):
    msgs = [
        {"role": "system", "content": "You are 'PropertyPandit' - a witty Indian AI who predicts house prices with expert accuracy and who starts a first conversation with Namaste and greetings 🏡📊. Read my data csv file and answer accordingly and dont use abusive or informal words. Also answer in formal, professional and dont use much commas, intead use ! or emojis to seperate the lines. Speak in Hinglish (mix of Hindi and English) with Bollywood, and Indian internet meme style. Keep answers funny, lively, and relatable while still giving the correct and accurate information from the dataset. Sprinkle emojis like 😂🙌🔥 where it fits. Your personality is energetic, street-smart, comedian, and aware of trendy Indian memes. When someone asks “Who are you?”, just give a friendly, medium length, polite and memes way answer — don’t reveal the whole context."},
        {"role": "user", "content": myquestion}
    ]
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=msgs
    )
    return response.choices[0].message.content


# ---- Streamlit UI ----
st.title("🏡 House Price Predictor & AI Assistant")

# --- Ask AI Section ---
st.subheader("🤖 Ask PropertyPandit AI")
user_question = st.text_input("Ask AI anything...")
if st.button("Ask AI"):
    if user_question.strip():
        ai_response = price_pandit(user_question)
        st.text_area("AI Response", ai_response, height=150)

# --- Prediction Section ---
st.subheader("📈 Predict House Price")

city = st.selectbox("City", sorted(data["city"].unique()))
location = st.selectbox("Location", sorted(data["location"].unique()))
area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=20000, step=50)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=20, step=1)
furnishing = st.selectbox("Furnishing", ["Furnished", "Unfurnished", "Semi-Furnished"])
age_years = st.number_input("Age of Property (Years)", min_value=0, max_value=100, step=1)

if st.button("Predict Price"):
    price = predict_price(city, location, area_sqft, bedrooms, furnishing, age_years)
    st.success(price)