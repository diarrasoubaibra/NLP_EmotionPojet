# Importation des packages principaux
import streamlit as st 
import altair as alt
import plotly.express as px 

from googletrans import Translator 

# Importation des packages pour l'analyse de données (EDA)
import pandas as pd 
import numpy as np 
from datetime import datetime

# Importation des utilitaires pour la gestion des modèles
import joblib 

# Chargement du pipeline de classification des émotions pré-entraîné
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_2024.pkl", "rb"))

# Importation des utilitaires pour le suivi des pages et des prédictions
from track_utils import (
    create_page_visited_table,
    add_page_visited_details,
    view_all_page_visited_details,
    add_prediction_details,
    view_all_prediction_details,
    create_emotionclf_table
)

# Fonction pour prédire les émotions à partir du texte donné
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Fonction pour obtenir les probabilités de prédiction pour chaque émotion
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Dictionnaire pour associer chaque émotion à un emoji
emotions_emoji_dict = {
    "colère": "😠", "dégoût": "🤮", "peur": "😨😱",
    "heureux": "🤗", "joie": "😂", "neutre": "😐",
    "triste": "😔", "tristesse": "😔", "honte": "😳", "surprise": "😮"
}

# Dictionnaire pour mapper les émotions en anglais à la version française
prediction_mapping = {
    "anger": "colère",
    "disgust": "dégoût",
    "fear": "peur",
    "happy": "heureux",
    "joy": "joie",
    "neutral": "neutre",
    "sad": "triste",
    "sadness": "tristesse",
    "shame": "honte",
    "surprise": "surprise"
}

# Initialisation du traducteur
translator = Translator()

# Fonction principale de l'application Streamlit
def main():
    # Titre de l'application
    st.title("Application de Classification des Émotions")
    
    # Création du menu latéral
    menu = ["Accueil", "Monitoring", "À Propos"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Création des tables pour le suivi des pages et des prédictions si elles n'existent pas
    create_page_visited_table()
    create_emotionclf_table()
    
    # Logique pour la page "Accueil"
    if choice == "Accueil":
        # Enregistrement de la visite de la page "Accueil"
        add_page_visited_details("Accueil", datetime.now())
        st.subheader("Accueil - Émotions dans le texte")

        # Formulaire pour saisir du texte
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Tapez ici")  # Zone de texte pour l'utilisateur
            submit_text = st.form_submit_button(label='Soumettre')  # Bouton pour soumettre le texte
            
        # Si l'utilisateur soumet le texte
        if submit_text:
            # Traduire le texte en anglais avant de faire la prédiction
            translated_text = translator.translate(raw_text, dest='en').text

            col1, col2 = st.columns(2)  # Création de deux colonnes pour afficher les résultats

            # Prédiction de l'émotion et obtention des probabilités
            prediction = predict_emotions(translated_text)
            probability = get_prediction_proba(translated_text)
            
            # Transformer la prédiction en français
            prediction_fr = prediction_mapping.get(prediction, prediction)
            emoji_icon = emotions_emoji_dict.get(prediction_fr, "🚫")

            # Enregistrement des détails de la prédiction dans la base de données
            add_prediction_details(raw_text, prediction_fr, np.max(probability), datetime.now())

            # Affichage des résultats dans la première colonne
            with col1:
                st.success("Texte Original")
                st.write(raw_text)

                st.success("Prédiction")
                st.write("{} : {}".format(prediction_fr, emoji_icon))
                st.write("Confiance : {}".format(np.max(probability)))

            # Affichage des probabilités de prédiction dans la deuxième colonne
            with col2:
                st.success("Probabilité de Prédiction")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["émotions", "probabilité"]

                # Graphique en barres des probabilités
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='émotions',
                    y='probabilité',
                    color='émotions'
                )
                st.altair_chart(fig, use_container_width=True)

    # Logique pour la page "Surveiller"
    elif choice == "Monitoring":
        # Enregistrement de la visite de la page "Surveiller"
        add_page_visited_details("Monitoring", datetime.now())
        st.subheader("Monitoring de l'Application")

        # Affichage des métriques des pages visitées
        with st.expander("Métriques de Pages"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Nom de Page', 'Heure de Visite'])
            st.dataframe(page_visited_details)

            # Graphique en barres du nombre de visites par page
            pg_count = page_visited_details['Nom de Page'].value_counts().rename_axis('Nom de Page').reset_index(name='Comptes')
            c = alt.Chart(pg_count).mark_bar().encode(x='Nom de Page', y='Comptes', color='Nom de Page')
            st.altair_chart(c, use_container_width=True)

            # Diagramme circulaire du nombre de visites par page
            p = px.pie(pg_count, values='Comptes', names='Nom de Page')
            st.plotly_chart(p, use_container_width=True)

        # Affichage des métriques du classificateur d'émotions
        with st.expander('Métriques du Classificateur d\'Émotions'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Texte Brut', 'Prédiction', 'Probabilité', 'Heure de Visite'])
            st.dataframe(df_emotions)

            # Graphique en barres du nombre de prédictions par émotion
            prediction_count = df_emotions['Prédiction'].value_counts().rename_axis('Prédiction').reset_index(name='Comptes')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prédiction', y='Comptes', color='Prédiction')
            st.altair_chart(pc, use_container_width=True)

    # Logique pour la page "À Propos"
    else:
        st.subheader("À Propos")
        # Enregistrement de la visite de la page "À Propos"
        add_page_visited_details("À Propos", datetime.now())

# Point d'entrée principal de l'application
if __name__ == '__main__':
    main()
