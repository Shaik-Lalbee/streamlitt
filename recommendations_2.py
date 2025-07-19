# recommendation_app.py
# Real-Time Laptop Recommendation Engine using Streamlit and Scikit-learn

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Laptop Dataset --------------------
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Laptop': [
            "Apple MacBook Air M2",
            "Apple MacBook Pro M3",
            "Dell XPS 13",
            "Dell Inspiron 15",
            "HP Spectre x360",
            "HP Pavilion 14",
            "Lenovo ThinkPad X1 Carbon",
            "Lenovo IdeaPad Flex 5",
            "Asus ZenBook 14",
            "Asus ROG Zephyrus G14",
            "Acer Swift 3",
            "MSI GF65 Thin",
        ],
        'Description': [
            "Apple's lightweight laptop with M2 chip, Retina display, long battery life",
            "High-performance MacBook with M3 chip, excellent for professionals",
            "Premium ultrabook with InfinityEdge display, Intel Evo platform",
            "Affordable Dell laptop with good specs for students and professionals",
            "HP's flagship 2-in-1 with OLED display and Intel Core i7 processor",
            "Budget-friendly laptop with Ryzen 5 processor and Full HD display",
            "Business laptop with carbon fiber build, great keyboard and battery",
            "Convertible laptop with touch display and AMD Ryzen 7 processor",
            "Slim and sleek design with powerful Intel Core i5 processor",
            "Gaming laptop with Ryzen 9, RTX 3060 graphics, 120Hz display",
            "Compact and fast ultrabook with Ryzen processor and fast charging",
            "MSI gaming laptop with NVIDIA graphics and advanced cooling",
        ]
    })

# -------------------- Recommendation Function --------------------
def get_recommendations(selected_laptop, df, top_n=3):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = df[df['Laptop'] == selected_laptop].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the selected laptop itself
    recommendations = [df['Laptop'].iloc[i[0]] for i in sim_scores]
    return recommendations

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Laptop Advisor", layout="centered")
st.title("🔍 Real-Time Laptop Recommendation Engine")
st.caption("Choose a laptop model and get intelligent suggestions for similar devices.")

# Load Data
df = load_data()

# User Selection
laptop_selected = st.selectbox("💻 Choose a Laptop", df['Laptop'].tolist())

if laptop_selected:
    st.markdown(f"### ✅ You selected: `{laptop_selected}`")
    recommended = get_recommendations(laptop_selected, df)
    
    st.markdown("### 🧠 Recommended Alternatives:")
    for i, rec in enumerate(recommended, start=1):
        st.write(f"**{i}. {rec}**")

# Expandable Full Dataset
with st.expander("📄 View Full Laptop Database"):
    st.dataframe(df)
