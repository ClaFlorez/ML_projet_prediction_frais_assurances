# README — Utilisation des bibliothèques et Évaluation du modèle

Ce document explique **l’utilisation de chaque bibliothèque** dans un flux typique de régression avec `scikit-learn` et présente en détail les **principales métriques d’évaluation**.

---

## 1) Exemple de code et utilisation des bibliothèques

```python
# ============================
# IMPORTS
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================
# DONNÉES D’EXEMPLE
# ============================
df = pd.DataFrame({
    "age":[19,23,31,45,52,60,21,47],
    "bmi":[27.9,23.1,31.2,28.5,33.1,29.4,24.2,30.8],
    "children":[0,1,2,2,3,0,1,2],
    "sex":["female","male","female","male","female","male","female","male"],
    "smoker":["yes","no","no","no","yes","no","no","yes"],
    "region":["southwest","southeast","northwest","northeast","southeast","southwest","northeast","northwest"],
    "charges":[16884.92,1826.84,46200.0,9782.9,44400.1,14000.0,2200.45,39000.0]
})

print(df.head())

# Visualisation exploratoire
plt.figure()
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker")
plt.title("Relation BMI vs Charges (color = fumeur)")
plt.tight_layout()
plt.show()

# ============================
# PRÉTRAITEMENT
# ============================
categorical_features = ["sex", "smoker", "region"]
numerical_features   = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)

poly = PolynomialFeatures(degree=2, include_bias=False)

# ============================
# PIPELINE COMPLET
# ============================
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("poly", poly),
    ("model", LinearRegression())
])

# ============================
# TRAIN / TEST SPLIT
# ============================
X = df.drop(columns=["charges"])
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ============================
# ENTRAÎNEMENT ET PRÉDICTION
# ============================
pipe.fit(X_train, y_train)
y_pred_train = pipe.predict(X_train)
y_pred_test  = pipe.predict(X_test)

# ============================
# ÉVALUATION
# ============================
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test  = mean_squared_error(y_test, y_pred_test)
rmse_train = np.sqrt(mse_train)
rmse_test  = np.sqrt(mse_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)

print(f"MSE  (train): {mse_train:.2f} | (test): {mse_test:.2f}")
print(f"RMSE (train): {rmse_train:.2f} | (test): {rmse_test:.2f}")
print(f"R²   (train): {r2_train:.4f} | (test): {r2_test:.4f}")

# Diagramme des résidus
residuals = y_test - y_pred_test
plt.figure()
sns.scatterplot(x=y_pred_test, y=residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Prédiction")
plt.ylabel("Résidu (y_true - y_pred)")
plt.title("Diagramme des résidus (jeu de test)")
plt.tight_layout()
plt.show()
```

### Utilisation des bibliothèques
- **NumPy (`np`)** : calculs numériques, racine carrée pour RMSE.  
- **Pandas (`pd`)** : création et manipulation de DataFrame.  
- **Matplotlib (`plt`)** : figures et graphiques de base.  
- **Seaborn (`sns`)** : visualisations statistiques avec style amélioré.  
- **`train_test_split`** : séparation apprentissage/test.  
- **`LinearRegression`** : modèle de régression linéaire.  
- **`PolynomialFeatures`** : expansion polynomiale des variables numériques.  
- **`OneHotEncoder`** : encodage des variables catégorielles.  
- **`mean_squared_error`, `r2_score`** : évaluation quantitative du modèle.  
- **`ColumnTransformer`** : application de transformations par type de colonne.  
- **`Pipeline`** : enchaînement du prétraitement et du modèle.  

---

## 2) Métriques d’évaluation de la régression

### 2.1. MSE — Mean Squared Error
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]
- Moyenne des carrés des erreurs.  
- Très sensible aux valeurs aberrantes.  
- Exprimé dans l’unité au carré de la variable cible.

### 2.2. RMSE — Root Mean Squared Error
\[
RMSE = \sqrt{MSE}
\]
- Racine carrée du MSE, donc dans la même unité que \(y\).  
- Plus interprétable que le MSE.  
- Comparez-le à l’écart-type de \(y\) pour juger la qualité du modèle.

### 2.3. MAE — Mean Absolute Error
\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]
- Moyenne des valeurs absolues des erreurs.  
- Plus robuste aux valeurs aberrantes.  
- Si RMSE >> MAE, cela indique la présence d’outliers.

```python
from sklearn.metrics import mean_absolute_error
mae_test = mean_absolute_error(y_test, y_pred_test)
```

### 2.4. R² — Coefficient de détermination
\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]
- Mesure la proportion de variance expliquée par le modèle.  
- Peut être négatif si le modèle est moins performant qu’une moyenne simple.  

**R² ajusté** corrige l’inflation artificielle liée au nombre de variables :
\[
R^2_{adj} = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}
\]

```python
p = pipe[:-1].transform(X_test).shape[1]
n = len(y_test)
r2_adj = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
```

### 2.5. MAPE — Mean Absolute Percentage Error
\[
MAPE = \frac{100}{n} \sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]
- Erreur en pourcentage, utile pour l’interprétation métier.  
- Attention : instable si \(y\) contient des valeurs proches de zéro.

### 2.6. Validation croisée
- Une seule séparation train/test peut être instable.  
- La validation croisée permet d’obtenir une estimation plus robuste.

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse_scorer(estimator, X, y):
    y_hat = estimator.fit(X, y).predict(X)
    return -np.sqrt(mean_squared_error(y, y_hat))

cv_scores = cross_val_score(pipe, X, y, cv=5, scoring=rmse_scorer)
print("RMSE CV (5 folds):", -cv_scores.mean(), "+/-", cv_scores.std())
```

### 2.7. Analyse des résidus
- Vérification des hypothèses : linéarité, homoscédasticité, indépendance, normalité approximative.  
- Graphiques utiles :  
  - résidus vs prédictions,  
  - histogramme ou QQ-plot des résidus.  

```python
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-plot des résidus (test)")
plt.show()
```

---

## 3) Points clés pour choisir les métriques
- Si les grandes erreurs coûtent cher → privilégier **RMSE/MSE**.  
- Si présence d’outliers → consulter aussi **MAE**.  
- Pour comparer modèles complexes → utiliser **R² ajusté**.  
- Pour interprétation métier → **MAPE** si valeurs de y pas proches de zéro.  
- Toujours valider avec **cross-validation** et analyser les résidus.  
