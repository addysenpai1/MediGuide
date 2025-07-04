import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import hashlib
import os 
import joblib 
import warnings
import sklearn
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# warnings.filters('ignore')
import pn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import cv2
import tensorflow as tf
from keras.models import load_model
import requests
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage


st.set_page_config(
    page_title="Unified Health Portal",
    page_icon="💊",
    layout="wide"
)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


USER_DB_FILE = "users.csv"
if not os.path.exists(USER_DB_FILE):
    df = pd.DataFrame(columns=["username", "password"])
    df.to_csv(USER_DB_FILE, index=False)

def load_users():
    return pd.read_csv(USER_DB_FILE)

def save_user(username, password):
    df = load_users()
    if username in df["username"].values:
        return False 
    new_user = pd.DataFrame({"username": [username], "password": [hash_password(password)]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB_FILE, index=False)
    return True

def authenticate(username, password):
    df = load_users()
    if username in df["username"].values:
        stored_password = df[df["username"] == username]["password"].values[0]
        return stored_password == hash_password(password)
    return False

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""

if st.session_state["authenticated"]:
    st.sidebar.title("Logout")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    st.title(f"Welcome, {st.session_state['username']}!")
    st.write("Explore the Unified Health Portal!")
    @st.cache_data
    def load_data():
        precaution = pd.read_csv("dataset/precautions_df.csv")
        workout = pd.read_csv("dataset/workout_df.csv")
        medication = pd.read_csv('dataset/medications.csv')
        diets = pd.read_csv('dataset/diets.csv')
        description = pd.read_csv("dataset/description.csv")
        return precaution, workout, medication, diets, description
    precaution, workout, medication, diets, description = load_data()
    medicines_dict = pickle.load(open('Model_Disease_predictor/medicine_dict.pkl', 'rb'))
    medicines = pd.DataFrame(medicines_dict)
    similarity = pickle.load(open('Model_Disease_predictor/similarity.pkl', 'rb'))
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory()
    def get_response(query):
        prompt = f"""Reply only if the query is related to disease.
                    If it is not related to that then reply with 'I can't understand the query'. 
                    Query: {query}
                    Note: Give only a short reply
                    Note: Answer in points
                    Format: 
                        query:       
                        answer: reply should contain minimum 3 lines
                    """
        _data = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=_data)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'response': f"We are not able to process the query ({response.status_code}): {response.text}"}
    
    def get_disease_details(disease):
        disease = disease.lower()
        descr = description[description['Disease'].str.lower() == disease]['Description']
        descr = descr.iloc[0] if not descr.empty else "No description available."
        pre = precaution[precaution['Disease'].str.lower() == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = pre.values.flatten().tolist() if not pre.empty else ["No precautions available."]
        die = diets[diets['Disease'].str.lower() == disease]['Diet']
        die = die.tolist() if not die.empty else ["No diet information available."]
        work = workout[workout['disease'].str.lower() == disease]['workout']
        work = work.tolist() if not work.empty else ["No workout information available."]
        med = medication[medication['Disease'].str.lower() == disease]['Medication']
        med = med.tolist() if not med.empty else ["No medication information available."]

        return {
            "description": descr,
            "precautions": pre,
            "diets": die,
            "medications": med,
            "workouts": work
        }
    def recommend(medicine):
        medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
        distances = similarity[medicine_index]
        medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_medicines = []
        for i in medicines_list:
            recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)
        return recommended_medicines
    st.markdown(
        """
        <style>
        .css-1d391kg {
            font-size: 26px !important;  /* Increased font size */
        }
        .css-1qk4v5z {
            font-size: 24px !important;  /* Adjusted size for specific items */
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.sidebar.title("MediGuide")
    menu = st.sidebar.radio(
        "Select an option:",
        ("Home","Disease Predictor","Disease Information", "Medicine Recommender System", "Health Chatbot", "Profile", "Help")
    )
    if menu == "Home":
        st.title("MEDIGUIDE")
        st.divider()
        col1, col2= st.columns(2)
        with col1:
            st.write("Welcome to MEDIGUIDE")
            st.write("we are here to help you regarding  all your queries related to health issuse")
            st.write("Users are requested to navigate to their required prediction")
            st.write("""THANK YOU""")

        with col2:
            st.image("images/bg.png", width=400)

    elif menu == "Disease Information":
        st.title("Disease Information Model")
        st.info("🚨 If you have any allergies or allgergy related medical condition, we recommend using the chatbot to personalize your recommendations ")

        diseases = ["Select a disease"] + [
            "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction",
            "Peptic ulcer disease", "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
            "Hypertension", "Migraine", "Cervical spondylosis", "Paralysis (brain hemorrhage)",
            "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid", "Hepatitis A",
            "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis",
            "Tuberculosis", "Common Cold", "Pneumonia", "Dimorphic hemorrhoids (piles)",
            "Heart attack", "Varicose veins", "Hypothyroidism", "Hyperthyroidism",
            "Hypoglycemia", "Osteoarthritis", "Arthritis", "(vertigo) Paroxysmal Positional Vertigo",
            "Acne", "Urinary tract infection", "Psoriasis", "Impetigo"
        ]

        disease = st.selectbox("Select the disease you want to know about:", diseases)
        if disease and disease != "Select a disease":
            if disease == "Allergy":
                st.header("Details for Allergy")
                st.subheader("Description")
                st.write("Allergies occur when the immune system reacts to substances in the environment that are usually harmless.")
                st.subheader("Precautions")
                st.write("- Avoid allergens (dust, pollen, pet dander, food triggers).")
                st.write("- Use anti-allergic medications if prescribed.")
                st.write("- Keep surroundings clean and free from allergens.")
                st.subheader("Diet Recommendations")
                st.write("- Consume Vitamin C-rich foods (oranges, lemons) to boost immunity.")
                st.write("- Drink plenty of water to flush out toxins.")
                st.write("- Avoid processed and high-histamine foods (fermented foods, alcohol, shellfish).")
                st.subheader("Workout Recommendations")
                st.write("- Engage in light exercises like walking or yoga.")
                st.write("- Avoid outdoor exercises during high pollen seasons.")
                st.subheader("Medications")
                st.write("- Antihistamines (Loratadine, Cetirizine)")
                st.write("- Nasal sprays for congestion")
                st.write("- EpiPen (for severe allergic reactions)")
                st.info("For severe allergic reactions, seek immediate medical help.")
            elif disease:
                details = get_disease_details(disease)
                st.header(f"Details for {disease.title()}")
                st.subheader("Description")
                st.write(details["description"])
                st.subheader("Precautions")
                st.write(", ".join(details["precautions"]))
                st.subheader("Diet Recommendations")
                st.write(", ".join(details["diets"]))
                st.subheader("Workout Recommendations")
                st.write(", ".join(details["workouts"]))
                st.subheader("Medications")
                st.write(", ".join(details["medications"]))
            else:
                st.info("Please select a disease to get started.")
    elif menu == "Medicine Recommender System":
        st.title("💊 Advanced Medicine Recommender System")
        medicine_options = ["Select a medicine"] + list(medicines['Drug_Name'].values)
        selected_medicine_name = st.selectbox(
            '🔍 Select the medicine whose alternative is to be recommended:',
            medicine_options,
            index=0,
            help="Start typing the medicine name and select from the dropdown."
        )
        if selected_medicine_name != "Select a medicine" and st.button('🚀 Recommend Medicine'):
            recommendations = recommend(selected_medicine_name)
            st.subheader(f"💊 Recommended Medicines for '{selected_medicine_name}':")

            for index, medicine in enumerate(recommendations, 1):
                st.write(f"✅ Recommendation {index}: {medicine}")
                st.markdown(f"[🛒 Purchase {medicine} on PharmEasy](https://pharmeasy.in/search/all?name={medicine})")
        else:
            st.info("Please select a medicine to get recommendations. ⚕️")
    elif menu == "Health Chatbot":
        user_input = st.chat_input("Ask me anything:", key="chat_input")
        if user_input:
            st.session_state["memory"].save_context({"user": user_input}, {"bot": ""})  
            response = "I'm sorry, I don't understand that query."
            if "disease" in user_input.lower():
                disease_name = user_input.lower().replace('disease', '').strip()
                details = get_disease_details(disease_name)
                response = f"You can use the 'Disease Information' section to learn more about specific diseases.\n\n"
                response += f"**Description:** {details['description']}\n\n"
                response += f"**Precautions:**\n- " + "\n- ".join(details['precautions']) + "\n\n"
                response += f"**Diet Recommendations:**\n- " + "\n- ".join(details['diets']) + "\n\n"
                response += f"**Workout Recommendations:**\n- " + "\n- ".join(details['workouts']) + "\n\n"
                response += f"**Medications:**\n- " + "\n- ".join(details['medications']) + "\n"
            elif "medicine" in user_input.lower():
                response = "To find alternative medicines, go to the 'Medicine Recommender System' section."
            elif "help" in user_input.lower():
                response = "I'm here to assist you! You can ask about how to navigate this portal or general health queries."
            else:
                llm_Response = get_response(user_input)
                response = llm_Response['response']
            st.session_state["memory"].chat_memory.add_message(AIMessage(content=response))
        st.subheader("Chat History")
        for msg in st.session_state["memory"].chat_memory.messages:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**You:** {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"**Bot:** {msg.content}")
        if st.button("Clear Chat"):
            st.session_state["memory"].clear()
    elif menu == "Disease Predictor":
            model = joblib.load("Model_Disease_predictor/xgboostmodel.pkl")
            model2 = joblib.load("Model_Disease_predictor/diabetes.pkl")
            model4 = joblib.load("Model_Disease_predictor/Prostate_Cancer.pkl")
            scaler = joblib.load('Model_Disease_predictor/scaler.pkl')
            rf_model = joblib.load('Model_Disease_predictor/random_forest_model.pkl')
            xgb_model = joblib.load('Model_Disease_predictor/xgboost_model.pkl')
            file_path = "dataset/symbipredict_2022.csv"
            df = pd.read_csv(file_path)
            def get_encoded_value(mapping, key):
                return mapping.get(key, key)
            navigation = st.sidebar.radio("Navigation", ["***DISEASE DETECTION***","SKIN DISEASE PREDICTOR"])
            if navigation =="***DISEASE DETECTION***":
                    st.title("Disease Detection")
                    st.button("Submit  disease data")
                    if 'df' not in locals() and 'df' not in globals():
                        st.error("Dataset not loaded. Please check the data loading process.")
                    else:
                        label_encoder = LabelEncoder()
                        df['prognosis_encoded'] = label_encoder.fit_transform(df['prognosis']) 
                        valid_symptoms = list(df.drop(columns=['prognosis', 'prognosis_encoded']).columns)       
                        def get_symptoms(disease):
                            """Retrieve symptoms associated with a disease."""
                            disease_rows = df[df['prognosis'] == disease]
                            symptom_sums = disease_rows.drop(columns=['prognosis', 'prognosis_encoded']).sum()
                            symptoms = symptom_sums[symptom_sums > 0].index.tolist()
                            return symptoms  
                        disease_symptoms = {disease: get_symptoms(disease) for disease in df['prognosis'].unique()}
                        def suggest_symptoms(input_symptoms):
                            """Suggest additional symptoms based on the closest matching disease."""
                            matched_disease = None
                            max_match = 0
                            suggested_symptoms = []   
                            for disease, symptoms in disease_symptoms.items():
                                match_count = len(set(input_symptoms) & set(symptoms))
                                if match_count > max_match:
                                    max_match = match_count
                                    matched_disease = disease
                                    suggested_symptoms = list(set(symptoms) - set(input_symptoms))
                            return matched_disease, suggested_symptoms
                        st.write("Select your symptoms and get a disease prediction.")
                        selected_symptoms = st.multiselect("Select symptoms:", valid_symptoms)
                        if 'enter_pressed' not in st.session_state:
                            st.session_state.enter_pressed = False
                        if st.button("ENTER"):
                            matched_disease, suggested_symptoms = suggest_symptoms(selected_symptoms)
                            if matched_disease:
                                st.write("**DO YOU HAVE ANY OF THE MENTIONED SYMPTOMS**")
                                st.write(suggested_symptoms if suggested_symptoms else "No additional symptoms to suggest.")
                            else:
                                st.write("No matching disease found.")
                            st.session_state.enter_pressed = True
                        if st.button("SUBMIT") and st.session_state.enter_pressed:
                            if selected_symptoms:
                                input_features = [1 if symptom in selected_symptoms else 0 for symptom in valid_symptoms]
                                if 'rf_model' not in globals() or 'xgb_model' not in globals():
                                    st.error("Error: Models not loaded. Please check model loading.")
                                else:
                                    prediction_rf = rf_model.predict([input_features])[0]
                                    prediction_xgb = xgb_model.predict([input_features])[0]
                                    disease_rf = label_encoder.inverse_transform([prediction_rf])[0]
                                    disease_xgb = label_encoder.inverse_transform([prediction_xgb])[0]
                                    st.success(f"** Prediction:** {disease_xgb}")
                                    if f"{disease_xgb}" == "Migraine":
                                        st.write("For further diagnostics, you can go to the following navigation: MIGRAINE HEADACHE")
                                    if f"{disease_xgb}" == "HeartAttack":
                                        st.write("For further diagnostics, you can go to the following navigation: HEART DISEASE")
                            else:
                                st.warning("Please select at least one symptom.")
                        else:
                            if not st.session_state.enter_pressed:
                                st.warning("Please click ENTER first before clicking SUBMIT.")

            elif navigation == "SKIN DISEASE PREDICTOR":
                model = load_model("Model_Disease_predictor/skin_disease_mobilenetv2.h5")
                class_labels = [
                    "BA-cellulitis", "BA-impetigo", "FU-athlete-foot", "FU-nail-fungus",
                    "FU-ringworm", "PA-cutaneous-larva-migrans", "VI-chickenpox", "VI-shingles"
                ]
                st.title("🩺 Skin Disease Predictor")
                st.write("Upload an image to detect the skin disease.")
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    if st.checkbox("Show Uploaded Image"):
                        st.image(image, caption="Uploaded Image", use_column_width=True)    
                    image = np.array(image)
                    image = cv2.resize(image, (224, 224))  
                    image = image / 255.0  
                    image = np.reshape(image, [1, 224, 224, 3])  
                    predictions = model.predict(image)
                    predicted_index = np.argmax(predictions[0]) 
                    predicted_label = class_labels[predicted_index]
                    confidence = predictions[0][predicted_index]
                    st.success(f"**Predicted Disease:** {predicted_label}")
                    st.info(f"**Confidence Score:** {confidence:.2f}")
                    st.info("BA:- Bacteria ,FU:- Fungus ,VI:- Virus ,PA:- Pathogens ")
    elif menu == "Profile":
        st.title("About Me")
        col1, col2 = st.columns([1, 3])
        with col1:
            image = Image.open('images/adarsh.jpg')  
            st.image(image, caption="MediGuide Developer", width=150)

        with col2:
            st.subheader("MediGuide Creator / Adarshkumar")
            st.write("Hi! I'm the developer of MediGuide, a platform aimed at empowering individuals with early disease prediction and health management tools.")

            st.markdown("**Connect with me:**")
            st.markdown("- [LinkedIn](www.linkedin.com/in/adarshkumar-gautam-09a593229)")
            st.markdown("- [GitHub](https://github.com/example)")
            st.markdown("- [Email](adarshgautam3234@gmail.com)")

        st.write("---") 
        col3, col4 = st.columns([1, 3])
        with col3:
            image = Image.open('images/siddhesh.jpg')  
            st.image(image, caption="MediGuide Developer", width=150)

        with col4:
            st.subheader("MediGuide Creator / Siddhesh")
            st.write("Hi! I'm the developer of MediGuide, a platform aimed at empowering individuals with early disease prediction and health management tools.")

            st.markdown("**Connect with me:**")
            st.markdown("- [LinkedIn](http://linkedin.com/in/siddhesh-hagawane-452263349)")
            st.markdown("- [GitHub](https://github.com/SiddheshHagawane06)")
            st.markdown("- [Email](siddhagawane09@gmail.com)")

        st.write("---")  

        col5, col6 = st.columns([1, 3])

        with col5:
            image = Image.open('images/aster .png') 
            st.image(image, caption="MediGuide Developer", width=150)

        with col6:
            st.subheader("MediGuide Creator / Aster")
            st.write("Hi! I'm the developer of MediGuide, a platform aimed at empowering individuals with early disease prediction and health management tools.")

            st.markdown("**Connect with me:**")
            st.markdown("- [LinkedIn](http://www.linkedin.com/in/aster-noronha-557b08250)")
            st.markdown("- [GitHub](hhttps://github.com/ASTERNORONHA)")
            st.markdown("- [Email](asternoronha13@gmail.com)")
    elif menu == "Help":
        st.title("Help & Information")

        st.subheader("About MediGuide")
        st.write("MediGuide is a unified health portal designed to empower individuals with advanced predictive algorithms for symptom assessment. It offers personalized medical advice, including precautions, medications, and lifestyle recommendations.")

        st.subheader("How to Use")
        st.write("1. Navigate through the sidebar to explore different features such as Disease Information, Medicine Recommender, Chatbot, Profile, and Help.")
        st.write("2. Input relevant data (disease name or medicine) in the respective sections to get tailored results.")
        st.write("3. Use the chatbot for quick health-related queries or navigation help.")

        st.subheader("Need Assistance?")
        st.write("Feel free to reach out to us via the Profile section for any support or feedback.")
else:
    menu = st.sidebar.radio("Select an option:", ("Login", "Signup"))
    
    if menu == "Signup":
        st.sidebar.subheader("Create a New Account")
        new_username = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password", type="password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password")
        
        if st.sidebar.button("Signup"):
            if new_password != confirm_password:
                st.sidebar.error("Passwords do not match!")
            elif save_user(new_username, new_password):
                st.sidebar.success("Account created successfully! Please log in.")
            else:
                st.sidebar.error("Username already exists. Choose a different one.")
    
    elif menu == "Login":
        st.sidebar.subheader("Login to Your Account")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if authenticate(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")
    
    st.warning("Please log in to access the application.")
    col1, col2, col3 = st.columns([1, 2, 1]) 

    with col2:  
        st.image("images/medi_guide.jpg", width=700)
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: flex;
                flex-direction: column;
            }
        </style>
        """,
        unsafe_allow_html=True
    )