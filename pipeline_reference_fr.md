# Référence rapide — `sklearn.pipeline` (Pipeline, make_pipeline, FeatureUnion, ColumnTransformer)

Ce document présente de manière concise et structurée les principales options et usages de la **pipeline** dans scikit-learn : création, méthodes, accès aux hyperparamètres, intégration avec la validation croisée, parallélisation de flux de features, et bonnes pratiques.

---

## 1) Concepts et objectifs

- **Pipeline** : enchaîne des **transformations** puis un **estimateur final** en un seul objet.
- **Avantages** : éviter les fuites de données (data leakage), garantir la reproductibilité, simplifier l'entraînement et la mise en production, faciliter l’optimisation d’hyperparamètres.

Schéma : `X ──> [transformateur 1] ──> [transformateur 2] ──> ... ──> [modèle] ──> ŷ`

---

## 2) Création d’une pipeline

### 2.1. `Pipeline`
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
```
- `steps` : liste de tuples `(nom, objet_sklearn)`.
- Le **dernier** objet doit être un estimateur (avec `fit` / `predict`). Les précédents doivent être des transformateurs (avec `fit` + `transform`).

### 2.2. `make_pipeline`
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipe = make_pipeline(StandardScaler(), LinearRegression())
```
- Noms d’étapes **générés automatiquement** (`standardscaler`, `linearregression`).
- Plus concis, mais moins de contrôle sur le nommage.

---

## 3) Combiner plusieurs flux de features

### 3.1. `ColumnTransformer` (recommandé pour colonnes hétérogènes)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=500))
])
```
- Applique différents traitements selon les colonnes. Idéal pour mélanger numériques/catégorielles.

### 3.2. `FeatureUnion` (concaténation parallèle de transformateurs)
```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

union = FeatureUnion([
    ("pca", PCA(n_components=5)),
    ("kbest", SelectKBest(score_func=f_classif, k=10))
])

pipe = Pipeline([("features", union), ("model", LogisticRegression())])
```
- Entraîne des branches **en parallèle**, concatène leurs sorties.
- Utile pour agréger plusieurs familles de représentations.

---

## 4) Méthodes essentielles d’une `Pipeline`

- `fit(X, y=None, **fit_params)` : ajuste tous les transformateurs puis le modèle final.
- `predict(X)` : applique `transform` successifs puis `predict` du modèle.
- `fit_transform(X, y=None)` : retourne les features transformées par la dernière étape **transformer** (ne passe pas au modèle).
- `score(X, y)` : délègue à `estimator.score`.
- `set_params(**params)` / `get_params(deep=True)` : accès aux hyperparamètres via la **double-underscore notation**.
  - Exemple : `pipe.set_params(model__C=1.0, preprocess__num__with_mean=False)`
- `named_steps` : dict donnant accès aux objets par leur nom (`pipe.named_steps["model"]`).

**Nomination des paramètres :** `"<nom_d_etape>__<nom_du_paramètre>"`

---

## 5) Intégration avec la validation croisée et la recherche d’hyperparamètres

### 5.1. `cross_val_score`
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")
print(-scores.mean(), scores.std())
```

### 5.2. `GridSearchCV` / `RandomizedSearchCV`
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "preprocess__cat__drop": [None, "first"],
    "model__C": [0.1, 1.0, 10.0]
}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X, y)

print(grid.best_params_)
best_model = grid.best_estimator_
```

**Remarques :**
- Les noms d’étapes entrent dans la clé du paramètre (`preprocess__cat__drop`).
- `n_jobs=-1` pour paralléliser si possible.

---

## 6) Options et paramètres utiles

- `memory=...` (dans `Pipeline`) : **cache** les résultats intermédiaires (utile avec cross-validation et gros prétraitements).
  ```python
  from joblib import Memory
  mem = Memory(location="cache_dir", verbose=0)
  pipe = Pipeline(steps=[...], memory=mem)
  ```

- `verbose=True` : affiche l’exécution des étapes lors de `fit` / `predict`.

- `remainder` (dans `ColumnTransformer`) : que faire des colonnes non listées (`"drop"` ou `"passthrough"`).

- `handle_unknown="ignore"` (dans `OneHotEncoder`) : indispensable pour robustesse en production si de nouvelles catégories apparaissent.

- `drop="first"` (dans `OneHotEncoder`) : évite la colinéarité parfaite (optionnel).

---

## 7) Imbrication et patterns utiles

### 7.1. Pipeline imbriquée dans un `ColumnTransformer`
```python
from sklearn.impute import SimpleImputer

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

pipe = Pipeline([("preprocess", preprocess), ("model", LogisticRegression())])
```

### 7.2. `FunctionTransformer` pour logique custom
```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log1p_safe(X):
    return np.log1p(np.clip(X, a_min=0, a_max=None))

num_pipe = Pipeline([("log1p", FunctionTransformer(log1p_safe)), ("scaler", StandardScaler())])
```

### 7.3. `TransformedTargetRegressor` (cible transformée) — complément
```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

reg = TransformedTargetRegressor(
    regressor=pipe,  # pipeline de features
    transformer=PowerTransformer(method="yeo-johnson")
)
reg.fit(X, y)
```

---

## 8) Bonnes pratiques et pièges courants

- **Tout prétraitement dépendant des données** doit être **dans** la Pipeline (imputation, normalisation, encodage).
- Utiliser `ColumnTransformer` pour gérer proprement colonnes numériques/catégorielles.
- Standardiser les numériques pour les modèles sensibles à l’échelle (régression linéaire, SVM).
- Toujours évaluer via **validation croisée** ; ne pas fiter hors pipeline.
- Notation `etape__param` pour `GridSearchCV` ; vérifier les chemins des paramètres.
- Attention aux formats denses vs creux (`sparse_output`) selon les encodeurs et estimateurs.
- `handle_unknown="ignore"` pour robustesse face aux nouvelles catégories en production.

---

## 9) Exemple complet récapitulatif

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", Ridge())
])

param_grid = {
    "preprocess__num__imputer__strategy": ["median", "mean"],
    "model__alpha": [0.1, 1.0, 10.0]
}

search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="neg_root_mean_squared_error")
search.fit(X, y)

print("Meilleurs paramètres:", search.best_params_)
best = search.best_estimator_
y_pred = best.predict(X_test)
```

---

## 10) Check-list rapide

- [ ] Toutes les transformations de données sont-elles **dans** la Pipeline ?  
- [ ] Les colonnes sont-elles bien séparées (num vs cat) via `ColumnTransformer` ?  
- [ ] Les paramètres de recherche sont-ils bien nommés `etape__param` ?  
- [ ] `handle_unknown="ignore"` activé pour les catégorielles encodées ?  
- [ ] `memory` utilisé si prétraitements coûteux et CV intensive ?  
- [ ] Évaluation faite via CV (et pas un seul split) ?  

---

Fin de la référence.
