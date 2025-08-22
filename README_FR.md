
# Projet de Régression des Frais d’Assurance — Documentation Technique (README)

Ce dépôt présente une implémentation orientée-objet d’un pipeline de **régression des frais d’assurance** à partir du dataset `insurance.csv`.  
Les méthodes de la classe `PredictreurAssurance` sont documentées **ligne par ligne**, avec les **formules mathématiques** associées lorsque pertinent.

## Table des matières
1. [Prérequis et données](#prérequis-et-données)  
2. [Classe `PredictreurAssurance` : vue d’ensemble](#classe-predictreurassurance--vue-densemble)  
3. [Méthode `nettoyer_donnees()`](#méthode-nettoyer_donnees) — nettoyage complet  
4. [Méthode `pretraiter_donnees()`](#méthode-pretraiter_donnees) — encodage & split 80/20  
5. [Méthode `entrainer_modele(degre=1)`](#méthode-entrainer_modeledegre1) — linéaire ou polynomiale  
6. [Méthode `predire()`](#méthode-predire) — prédictions sur le test  
7. [Méthode `evaluer()`](#méthode-evaluer) — MSE, RMSE, R² et scatter  
8. [Méthode `visualiser()`](#méthode-visualiser) — graphiques complémentaires  
9. [Méthode `diagnostiquer()`](#méthode-diagnostiquer) — tableau de bord rapide  
10. [Annexe : corrélations (optionnel)](#annexe--corrélations-optionnel)  

---

## Prérequis et données

- Python 3.10+ recommandé
- Bibliothèques : `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Données : `insurance.csv` (hébergé sur Kaggle, lien « raw » GitHub possible)

Exemple de chemin GitHub (mode *raw*) :
```
https://raw.githubusercontent.com/ClaFlorez/ML_projet_prediction_frais_assurances/main/insurance.csv
```

---

## Classe `PredictreurAssurance` : vue d’ensemble

La classe encapsule les étapes suivantes :
- Chargement du CSV dans `self.assurances`
- Nettoyage : conversions de types, NA critiques, doublons → `self.df_clean` et `self.data_complet`
- Prétraitement : encodage One-Hot des catégorielles, séparation train/test (80/20)
- Entraînement : régression linéaire ou polynomiale
- Prédiction, évaluation et visualisation

Exemple d’usage minimal :
```python
df_predictor = PredictreurAssurance(PATH_ASSURANCES)
df_predictor.nettoyer_donnees()
df_predictor.pretraiter_donnees()
df_predictor.entrainer_modele(degre=1)
df_predictor.diagnostiquer()
```

---

## Méthode `nettoyer_donnees()`

### Rôle
Préparer un jeu de données propre et cohérent. Au terme de cette méthode, `self.df_clean` contient une copie nettoyée de `self.assurances` et `self.data_complet` peut pointer vers la même version.

### Code canonique (extrait simplifié)
```python
def nettoyer_donnees(self):
    self._convertir_types()
    self._gerer_valeurs_manquantes()
    self._supprimer_doublons()
    self.df_clean = self.assurances.copy()
    self.data_complet = self.df_clean.copy()
```

### Explication ligne par ligne
1. `self._convertir_types()`  
   - Convertit les colonnes numériques en types adaptés. Par exemple :
     ```python
     self.assurances['age'] = pd.to_numeric(self.assurances['age'], errors='coerce').astype('Int64')
     self.assurances['bmi'] = pd.to_numeric(self.assurances['bmi'], errors='coerce').astype('float64')
     self.assurances['charges'] = pd.to_numeric(self.assurances['charges'], errors='coerce').astype('float64')
     self.assurances['children'] = pd.to_numeric(self.assurances['children'], errors='coerce').astype('Int64')
     ```
   - `errors='coerce'` force les valeurs invalides en `NaN` (qui seront gérées ensuite).  
   - Les colonnes catégorielles (`sex`, `smoker`, `region`) sont normalisées (`str.strip()`, remplacement de `nan` textuels par `pd.NA`).

2. `self._gerer_valeurs_manquantes()`  
   - Supprime les lignes avec `NaN` dans **les colonnes critiques** : `age`, `smoker`, `bmi`, `charges`.
   - Raison : `charges` est la **cible** et doit être observée; `age`, `smoker`, `bmi` sont des *features* majeures.
   - Après suppression, la taille du DataFrame est affichée.

3. `self._supprimer_doublons()`  
   - Élimine les lignes dupliquées via `drop_duplicates()`.
   - Affiche le nombre de lignes supprimées.

4. `self.df_clean = self.assurances.copy()`  
   - Fige une **copie propre** pour l’EDA et le prétraitement.

5. `self.data_complet = self.df_clean.copy()`  
   - Pointeur pratique pour la suite (peut rester identique à `df_clean` à ce stade).

---

## Méthode `pretraiter_donnees()`

### Rôle
Transformer les variables catégorielles en indicatrices (One-Hot) et séparer les données en **train/test**.

### Code canonique (extrait)
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def pretraiter_donnees(self, test_size=0.2, random_state=42):
    X = self.df_clean.drop("charges", axis=1)
    y = self.df_clean["charges"]

    colonnes_cat = ["sex", "smoker", "region"]
    colonnes_existantes = [c for c in colonnes_cat if c in X.columns]

    self.preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), colonnes_existantes)
        ],
        remainder="passthrough"
    )

    X_transforme = self.preprocessor.fit_transform(X)

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X_transforme, y, test_size=test_size, random_state=random_state
    )
```

### Explication ligne par ligne
- `X = self.df_clean.drop("charges", axis=1)` et `y = self.df_clean["charges"]`  
  - Séparent **features** et **cible**.  
  - Notation : soit \( X \in \mathbb{R}^{n \times p} \), \( y \in \mathbb{R}^n \).

- `colonnes_existantes`  
  - Filtre des colonnes catégorielles présentes afin d’éviter les erreurs si une colonne manque.

- `ColumnTransformer(...)` avec `OneHotEncoder(drop="first", handle_unknown="ignore")`  
  - Encodage One-Hot : pour une variable catégorielle \( C \) à \( K \) modalités \( \{c_1,\dots,c_K\} \), on crée \( K-1 \) colonnes indicatrices  
    \[
    \phi_k(C) = \mathbb{1}[C = c_k], \quad k=1,\dots,K-1.
    \]  
  - `drop="first"` retire une colonne de référence pour éviter la **multicolinéarité parfaite** (piège des variables fictives).  
  - `handle_unknown="ignore"` évite une erreur si une catégorie nouvelle apparaît en prédiction.  
  - `remainder="passthrough"` conserve les colonnes numériques telles quelles.

- `X_transforme = self.preprocessor.fit_transform(X)`  
  - Apprend l’encodage sur le **train** et transforme \( X \) en matrice numérique \( \tilde{X} \).

- `train_test_split(..., test_size=0.2, ...)`  
  - Partitionne \( \tilde{X} \) et \( y \) en 80 % **train** et 20 % **test**.  
  - Notation : \( (X_{\text{train}}, y_{\text{train}}), (X_{\text{test}}, y_{\text{test}}) \).

---

## Méthode `entrainer_modele(degre=1)`

### Rôle
Ajuster un modèle de régression linéaire (degré 1) ou polynomiale (degré \( d>1 \)).

### Code canonique (extrait)
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def entrainer_modele(self, degre=1):
    if degre == 1:
        self.modele = LinearRegression()
    else:
        self.modele = Pipeline([
            ("poly", PolynomialFeatures(degree=degre, include_bias=False)),
            ("linreg", LinearRegression())
        ])
    self.modele.fit(self.X_train, self.y_train)
```

### Explication et formules
- **Régression linéaire**  
  - Modèle : \( \hat{y} = X \beta \).  
  - Estimateur des **moindres carrés ordinaires** (MCO) :  
    \[
    \hat{\beta} = (X^\top X)^{-1} X^\top y,
    \]  
    lorsque \( X^\top X \) est inversible.

- **Régression polynomiale** (via `PolynomialFeatures`)  
  - Extension des features : pour \( x = (x_1,\dots,x_p) \), on crée toutes les combinaisons jusqu’au degré \( d \) (sans le biais si `include_bias=False`).  
  - Exemple \( p=2, d=2 \) : \( [x_1, x_2, x_1^2, x_1x_2, x_2^2] \).  
  - Le modèle reste **linéaire en les paramètres** mais non linéaire en les **features** transformées.

- `self.modele.fit(self.X_train, self.y_train)`  
  - Ajuste les paramètres \(\hat{\beta}\) (ou du pipeline) sur l’ensemble d’entraînement.

---

## Méthode `predire()`

### Rôle
Produire les prédictions du modèle sur le jeu de test par défaut.

### Code canonique (extrait)
```python
def predire(self, X=None):
    if self.modele is None:
        raise ValueError("Modèle non entraîné.")
    if X is None:
        X = self.X_test
    y_pred = self.modele.predict(X)
    return y_pred
```

### Explication
- Contrôles d’usage : modèle entraîné et données de test disponibles.  
- `self.modele.predict(X)` calcule \(\hat{y}\) :  
  \[
  \hat{y} = X \hat{\beta} \quad \text{(ou } \hat{y} = \Phi(X)\hat{\beta} \text{ si polynômial)}
  \]
  où \(\Phi(\cdot)\) désigne la transformation polynomiale si utilisée.

---

## Méthode `evaluer()`

### Rôle
Calculer **MSE**, **RMSE**, **R²** et tracer le **scatter** \( y_{\text{test}} \) vs \(\hat{y}\).

### Code canonique (extrait)
```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def evaluer(self, afficher_graphique=True, retourner_scores=True):
    y_pred = self.modele.predict(self.X_test)
    mse = mean_squared_error(self.y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(self.y_test, y_pred)
    if afficher_graphique:
        plt.figure(figsize=(6,5))
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        m = min(self.y_test.min(), y_pred.min())
        M = max(self.y_test.max(), y_pred.max())
        plt.plot([m, M], [m, M], linestyle="--")
        plt.xlabel("Valeurs réelles (y_test)"); plt.ylabel("Prédictions (y_pred)")
        plt.title("Prédictions vs Valeurs réelles"); plt.tight_layout(); plt.show()
    if retourner_scores:
        return {"mse": mse, "rmse": rmse, "r2": r2}
```

### Formules
- **MSE** (erreur quadratique moyenne) :  
  \[
  \mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2.
  \]
- **RMSE** :  
  \[
  \mathrm{RMSE} = \sqrt{\mathrm{MSE}}.
  \]
- **\(R^2\)** (coefficient de détermination) :  
  \[
  R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2},
  \]
  où \( \bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i \).

- **Scatter** et diagonale idéale** : la ligne en pointillé \( y = x \) indique des prédictions parfaites.

---

## Méthode `visualiser()`

### Rôle
Produire des visualisations complémentaires : scatter \( y_{\text{test}} \) vs \(\hat{y}\) ou distribution des résidus.

### Code canonique (extrait)
```python
def visualiser(self, type_plot="scatter"):
    y_pred = self.modele.predict(self.X_test)
    if type_plot == "scatter":
        # scatter y_test vs y_pred + diagonale
        ...
    elif type_plot == "residus":
        residus = self.y_test - y_pred
        # histogramme des résidus
        ...
```

### Explications
- **Résidus** : \( r_i = y_i - \hat{y}_i \). Une distribution centrée autour de 0 est souhaitable.  
- L’analyse visuelle complète les métriques quantitatives.

---

## Méthode `diagnostiquer()`

### Rôle
Fournir un tableau de bord rapide : métriques + scatter + histogramme des résidus.

### Code canonique (extrait)
```python
def diagnostiquer(self, afficher_scores=True):
    y_pred = self.modele.predict(self.X_test)
    residus = self.y_test - y_pred
    mse = mean_squared_error(self.y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(self.y_test, y_pred)
    # scatter y_test vs y_pred + diagonale
    # histogramme des résidus
    return {"mse": mse, "rmse": rmse, "r2": r2}
```

### Interprétation
- **MSE/RMSE** bas et **\(R^2\)** élevé indiquent une meilleure performance.  
- Le scatter proche de la diagonale et des résidus centrés autour de 0 suggèrent un modèle bien calibré.

---

## Annexe : corrélations (optionnel)

On peut créer des variables binaires simples pour `smoker` et `sex` et calculer la matrice de corrélation de Pearson.  
Formule de la corrélation de Pearson entre variables \(X\) et \(Y\) :  
\[
\rho_{X,Y} = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2}\, \sqrt{\sum_{i=1}^{n}(Y_i - \bar{Y})^2}}.
\]

---

## Bonnes pratiques
- Vérifier la présence d’outliers et la stabilité des coefficients (Ridge/Lasso en cas de multicolinéarité).
- Utiliser une validation croisée pour choisir le degré polynomial.
- Sérialiser le modèle (ex. `joblib`) pour une utilisation ultérieure.

