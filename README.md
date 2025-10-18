# Rapport Synthétique : Analyse et Prédiction du Temps de Livraison

## Introduction

Ce rapport présente une synthèse de l'analyse et des résultats obtenus à partir du notebook Jupyter `pasted_file_GXuJXN_Notebook.ipynb`. L'objectif principal de ce projet est de construire un modèle de machine learning capable de prédire le temps de livraison (`Delivery_Time_min`) en se basant sur un ensemble de données de livraisons.

## Démarche Méthodologique

Le notebook suit une approche structurée en plusieurs étapes, typiques d'un projet de science des données :

1.  **Importation des Bibliothèques** : Les bibliothèques Python essentielles pour la manipulation des données (`pandas`), la visualisation (`seaborn`, `matplotlib`), le prétraitement (`sklearn.preprocessing`) et la modélisation (`sklearn.model_selection`, `sklearn.ensemble`, `sklearn.svm`, `sklearn.linear_model`, `sklearn.metrics`) ont été importées.
2.  **Chargement et Exploration des Données** : Le jeu de données a été chargé à partir du fichier `MyRealData.csv`. Une exploration initiale a été réalisée à l'aide des méthodes `describe()` et `info()` pour comprendre la structure des données, identifier les types de variables (numériques, catégorielles) et obtenir des statistiques descriptives.
3.  **Prétraitement des Données** :
    *   **Gestion des Valeurs Manquantes** : Les valeurs manquantes ont été traitées en imputant la moyenne pour les colonnes numériques et le mode pour les colonnes catégorielles.
    *   **Gestion des Doublons** : Une vérification a été effectuée pour identifier et supprimer les lignes dupliquées, assurant ainsi l'intégrité de l'ensemble de données.
    *   **Suppression des Colonnes Non Pertinentes** : Les colonnes d'identification (`Order_ID`) ont été supprimées car elles n'apportent pas d'information prédictive.
    *   **Normalisation des Caractéristiques Numériques** : Les caractéristiques numériques ont été normalisées à l'aide de `MinMaxScaler` pour les ramener à une échelle commune (entre 0 et 1), ce qui est crucial pour la performance de certains algorithmes de machine learning.
    *   **Encodage des Variables Catégorielles** : Les variables catégorielles (par exemple, `Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`) ont été converties en représentations numériques via l'encodage One-Hot (`OneHotEncoder`).
4.  **Analyse Exploratoire et Visualisation** : Des visualisations ont été générées pour comprendre la distribution de la variable cible (`Delivery_Time_min`), les relations entre les différentes caractéristiques et la corrélation entre les variables numériques (via une heatmap).
5.  **Sélection de Caractéristiques** : Une étape de sélection de caractéristiques a été mise en œuvre (en utilisant `SelectKBest` avec `f_regression`) pour identifier les variables les plus influentes sur le temps de livraison, réduisant ainsi la dimensionalité et améliorant potentiellement la performance du modèle.
6.  **Modélisation et Évaluation** : Plusieurs modèles de régression ont été entraînés et évalués :
    *   `RandomForestRegressor`
    *   `SVR` (Support Vector Regressor)

Les modèles ont été entraînés sur un ensemble de données d'entraînement et leurs performances ont été mesurées sur un ensemble de test à l'aide de métriques clés telles que le **R2 score**, l'**Erreur Absolue Moyenne (MAE)** et l'**Erreur Quadratique Moyenne (MSE)**. Le notebook inclut également l'utilisation de `GridSearchCV` pour l'optimisation des hyperparamètres, ce qui indique une recherche des meilleures configurations pour les modèles.

## Résultats et Conclusions

Le notebook démontre une pipeline complète pour la prédiction du temps de livraison. Les étapes de prétraitement rigoureuses, l'analyse exploratoire des données et la comparaison de plusieurs modèles de régression sont des points forts du projet.