import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# Configuration de la page
st.set_page_config(page_title="IBM HR Dashboard", layout="wide")

# URL de l'API (à adapter une fois déployée sur Hugging Face)
API_URL = "https://ton-espace-api.hf.space/predict"
DATA_URL = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/ibm_hr_attrition.xlsx"

@st.cache_data
def load_data():
    # Chargement des données IBM
    df = pd.read_excel(DATA_URL)
    return df

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Aller vers", ["Dashboard Global", "Recherche Employé", "Simulation Attrition"])

# --- OPTION 1 : DASHBOARD GLOBAL ---
if selection == "Dashboard Global":
    st.title("📊 Statistiques Globales - Attrition IBM")
    
    col1, col2, col3 = st.columns(3)
    attrition_rate = (df['Attrition'] == 'Yes').mean()
    col1.metric("Taux d'Attrition", f"{attrition_rate:.1%}")
    col2.metric("Effectif Total", len(df))
    col3.metric("Salaire Mensuel Moyen", f"{df['MonthlyIncome'].mean():.0f} $")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.pie(df, names='Attrition', title="Proportion d'Attrition", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x="JobRole", color="Attrition", barmode="group", title="Attrition par métier")
        st.plotly_chart(fig2, use_container_width=True)

# --- OPTION 2 : RECHERCHE EMPLOYÉ ---
elif selection == "Recherche Employé":
    st.title("🔍 Profil de l'employé")
    emp_id = st.number_input("Entrez l'ID de l'employé (EmployeeNumber)", min_value=1)
    
    # On cherche l'employé dans le dataframe
    employee = df[df['EmployeeNumber'] == emp_id]
    
    if not employee.empty:
        st.success(f"Données trouvées pour l'employé {emp_id}")
        st.dataframe(employee.T) # Affichage vertical pour plus de lisibilité
    else:
        st.warning("Aucun employé trouvé avec cet ID.")

# --- OPTION 3 : FORMULAIRE DE PRÉDICTION ---
elif selection == "Simulation Attrition":
    st.title("🤖 Prédire un départ")
    st.info("Remplissez les informations pour interroger l'API MLflow.")

    with st.form("prediction_form"):
        # On crée des colonnes pour que le formulaire ne soit pas trop long
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 65, 30)
            daily_rate = st.number_input("DailyRate", value=800)
            distance = st.slider("DistanceFromHome", 1, 30, 5)
            education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
            travel = st.selectbox("BusinessTravel", df['BusinessTravel'].unique())
            
        with col2:
            job_level = st.selectbox("JobLevel", [1, 2, 3, 4, 5])
            income = st.number_input("MonthlyIncome", value=5000)
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            stock = st.selectbox("StockOptionLevel", [0, 1, 2, 3])
            dept = st.selectbox("Department", df['Department'].unique())

        with col3:
            role = st.selectbox("JobRole", df['JobRole'].unique())
            marital = st.selectbox("MaritalStatus", df['MaritalStatus'].unique())
            years_at_co = st.slider("YearsAtCompany", 0, 40, 5)
            environment = st.slider("EnvironmentSatisfaction", 1, 4, 3)

        # Bouton d'envoi
        submit = st.form_submit_button("Lancer la prédiction")

    if submit:
        # ⚠️ CRUCIAL : Le payload doit contenir TOUTES les colonnes attendues par ton API (Pydantic model)
        # On utilise une ligne du dataset original comme base pour remplir les champs non présents dans le formulaire
        payload = df.iloc[0].to_dict() 
        
        # On met à jour le dictionnaire avec les valeurs du formulaire
        form_data = {
            "Age": age, "BusinessTravel": travel, "DailyRate": daily_rate, "Department": dept,
            "DistanceFromHome": distance, "Education": education, "JobLevel": job_level,
            "JobRole": role, "MaritalStatus": marital, "MonthlyIncome": income,
            "OverTime": overtime, "StockOptionLevel": stock, "YearsAtCompany": years_at_co,
            "EnvironmentSatisfaction": environment
        }
        payload.update(form_data)
        
        # Suppression de la cible si elle est dans le dictionnaire
        payload.pop('Attrition', None)

        try:
            with st.spinner("Appel de l'API en cours..."):
                response = requests.post(API_URL, json=payload)
                result = response.json()
            
            if result["prediction"] == 1:
                st.error("⚠️ Risque d'attrition détecté !")
            else:
                st.success("✅ L'employé semble vouloir rester.")
        except Exception as e:
            st.error(f"Erreur lors de la connexion à l'API : {e}")