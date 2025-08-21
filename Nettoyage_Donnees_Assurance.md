# 🧹 Nettoyage et Préparation des Données (Data Cleaning)

Ce document décrit les étapes nécessaires pour nettoyer et préparer le
dataset **insurance.csv** en vue d'un projet de Machine Learning
(régression des frais d'assurance médicale).

------------------------------------------------------------------------

## 🔹 1. Vérifier les types de données

``` python
# Vérification des types
df.dtypes
```

**Explication :**\
- `age`, `bmi`, `children`, `charges` doivent être **numériques**.\
- `sex`, `smoker`, `region` sont **catégorielles** (type `object`).\
👉 Si une colonne est au mauvais type, utilisez `astype` pour corriger.

------------------------------------------------------------------------

## 🔹 2. Vérifier les valeurs manquantes

``` python
# Valeurs manquantes
df.isnull().sum()
```

**Explication :**\
- Le dataset original n'a normalement **aucune valeur manquante**.\
- Si des valeurs sont manquantes :\
- Option 1 : les supprimer (`dropna`).\
- Option 2 : les remplacer (`fillna`) avec moyenne, médiane ou mode.

------------------------------------------------------------------------

## 🔹 3. Vérifier les doublons

``` python
# Vérification des doublons
df.duplicated().sum()
```

**Explication :**\
- Si des doublons existent, on les supprime :

``` python
df = df.drop_duplicates()
```

------------------------------------------------------------------------

## 🔹 4. Encodage des variables catégorielles

Les colonnes `sex`, `smoker`, `region` doivent être converties en
variables numériques.

``` python
# Encodage One-Hot
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.head()
```

**Explication :**\
- `drop_first=True` évite la multicolinéarité.\
- Exemples :\
- `sex` → devient `sex_male` (1 si homme, 0 si femme).\
- `smoker` → devient `smoker_yes`.\
- `region` → devient 3 colonnes (`region_northwest`, `region_southeast`,
`region_southwest`).

------------------------------------------------------------------------

## 🔹 5. Normalisation / Standardisation (optionnelle)

Certaines variables numériques doivent être mises à la même échelle.

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cols_to_scale = ['age', 'bmi', 'children']
df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])
```

**Explication :**\
- On standardise `age`, `bmi` et `children`.\
- `charges` (variable cible) reste inchangée.

------------------------------------------------------------------------

## 🔹 6. Séparer Features et Cible

``` python
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]
```

**Explication :**\
- `X` = variables explicatives (features).\
- `y` = variable cible (`charges`).

------------------------------------------------------------------------

# ✅ Résumé du nettoyage

1.  Vérification des types de données.\
2.  Contrôle des valeurs manquantes et doublons.\
3.  Encodage One-Hot des variables catégorielles.\
4.  Standardisation des variables numériques (optionnel).\
5.  Séparation en **features (X)** et **target (y)**.
