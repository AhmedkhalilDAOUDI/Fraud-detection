# 🔍 Fraud Detection System

Détection automatique de transactions bancaires frauduleuses par apprentissage automatique.

## 🚀 Application déployée

**Streamlit: [fraud-detection-khalil.streamlit.app](https://fraud-detection-khalil.streamlit.app/)**
**Render: https://fraud-detection-2zh6.onrender.com**

## 📋 Description du projet

Ce projet implémente un pipeline ML complet pour la détection de fraudes sur le dataset
Credit Card Fraud Detection (Kaggle). Le dataset contient 284 807 transactions européennes
de septembre 2013, dont seulement 0.17% sont frauduleuses — un cas classique de
déséquilibre sévère des classes.

**Tâche ML :** Classification binaire (fraude / légitime)  
**Défi principal :** Déséquilibre sévère → 99.83% légitime, 0.17% fraude  
**Métrique principale :** PR-AUC (Precision-Recall AUC)

## 📊 Dataset

- **Source :** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Taille :** 284 807 transactions, 31 features
- **Features :** Time, Amount, V1-V28 (composantes PCA anonymisées)
- **Target :** Class (0 = légitime, 1 = fraude)

## 🔧 Pipeline ML

1. **EDA** — Analyse exploratoire, distribution des classes, corrélations
2. **Preprocessing** — Split stratifié 80/20, StandardScaler sur Time et Amount, SMOTE sur train uniquement
3. **Modélisation** — 13 algorithmes testés et comparés par PR-AUC
4. **Tuning** — RandomizedSearchCV sur les 3 meilleurs modèles candidats
5. **Évaluation finale** — Optimisation du threshold, matrice de confusion
6. **Déploiement** — Application Streamlit interactive

## 🏆 Résultats

### Comparaison des modèles (Top 5)

| Modèle          | PR-AUC     | ROC-AUC | Recall | Precision | F1     |
| --------------- | ---------- | ------- | ------ | --------- | ------ |
| **Extra Trees** | **0.8808** | 0.9805  | 0.8469 | 0.9326    | 0.8877 |
| Random Forest   | 0.8759     | 0.9658  | 0.8367 | 0.8283    | 0.8325 |
| CatBoost        | 0.8499     | 0.9830  | 0.8571 | 0.5793    | 0.6914 |
| XGBoost         | 0.8462     | 0.9783  | 0.8571 | 0.4641    | 0.6022 |
| LightGBM        | 0.8110     | 0.9520  | 0.8673 | 0.4913    | 0.6273 |

### Modèle final — Extra Trees (Threshold optimisé : 0.5767)

| Métrique  | Valeur |
| --------- | ------ |
| PR-AUC    | 0.8808 |
| ROC-AUC   | 0.9805 |
| Recall    | 0.8469 |
| Precision | 0.9326 |
| F1        | 0.8877 |

### Matrice de confusion

|                   | Prédit Légitime | Prédit Fraude |
| ----------------- | --------------- | ------------- |
| **Réel Légitime** | 56858           | 6             |
| **Réel Fraude**   | 15              | 83            |

- 83 fraudes détectées sur 98
- 6 fausses alarmes sur 56 864 transactions légitimes

## 🖥️ Application Streamlit

L'application permet de :

- Saisir les 30 features d'une transaction
- Obtenir la probabilité de fraude en temps réel
- Voir la classification avec le niveau de confiance

## 📁 Structure du projet

fraud-detection/
├── data/ # Dataset (non inclus — voir Kaggle)
├── models/ # Modèles sérialisés (Google Drive)
├── notebooks/
│ └── notebook.ipynb # Pipeline ML complet
├── app.py # Application Streamlit
├── requirements.txt # Dépendances
└── README.md

## ⚙️ Installation locale

```bash
git clone https://github.com/AhmedkhalilDAOUDI/Fraud-detection.git
cd Fraud-detection
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Stack technique

- **ML :** scikit-learn, XGBoost, LightGBM, CatBoost
- **Imbalance :** imbalanced-learn (SMOTE)
- **Déploiement :** Streamlit Cloud
- **Sérialisation :** joblib

## 👤 Auteur

Ahmed Khalil DAOUDI — MSDE (Edition 7) — EHTP  
Projet Module 5 : Machine Learning / Pr. Abdelhamid Fadil
