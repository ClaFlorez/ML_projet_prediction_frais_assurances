# 📂 Méthodes pour Accéder au Dataset `insurance.csv` dans Google Colab ou Jupyter

Ce document décrit plusieurs façons de charger le fichier
**insurance.csv** pour votre projet de prédiction des frais d'assurance
médicale.

------------------------------------------------------------------------

## 🔹 1. Utiliser `sample_data` (ou `/content/`)

**Description :**\
Copier ou télécharger directement le fichier dans le répertoire
temporaire de Colab.

**Code :**

``` python
import pandas as pd

df = pd.read_csv("/content/sample_data/insurance.csv")
df.head()
```

**Avantages :** - Simple et rapide.

**Inconvénients :** - ❌ Le fichier est supprimé quand la session Colab
est fermée.\
- ❌ Il faut recharger à chaque nouvelle session.

------------------------------------------------------------------------

## 🔹 2. Monter Google Drive

**Description :**\
Connecter Google Drive à Colab et charger le fichier depuis `/MyDrive/`.

**Code :**

``` python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/datasets/insurance.csv")
df.head()
```

**Avantages :** - ✅ Les fichiers restent sauvegardés.\
- ✅ Accessible depuis n'importe quel notebook.

**Inconvénients :** - ⚠️ Nécessite d'autoriser l'accès à chaque
session.\
- ⚠️ L'accès peut être un peu plus lent.

------------------------------------------------------------------------

## 🔹 3. Télécharger automatiquement via l'API Kaggle

**Description :**\
Configurer l'API Kaggle avec `kaggle.json` et télécharger le dataset
directement.

**Code :**

``` python
!pip install -q kaggle

# Télécharger le dataset Kaggle
!kaggle datasets download -d mirichoi0218/insurance -p .
!unzip -o insurance.zip

import pandas as pd
df = pd.read_csv("insurance.csv")
df.head()
```

**Avantages :** - ✅ Toujours la dernière version du dataset.\
- ✅ Idéal si vous utilisez plusieurs datasets Kaggle.

**Inconvénients :** - ⚠️ Configuration initiale requise
(`kaggle.json`).\
- ⚠️ Le fichier est supprimé si non sauvegardé dans Drive.

------------------------------------------------------------------------

## 🔹 4. Héberger sur GitHub (recommandé pour partager)

**Description :**\
Uploader le dataset dans un dépôt GitHub public et utiliser le lien brut
(Raw).

**Code :**

``` python
import pandas as pd

url = "https://raw.githubusercontent.com/<USER>/<REPO>/main/data/insurance.csv"
df = pd.read_csv(url)
df.head()
```

**Avantages :** - ✅ Pas besoin de Drive ni d'API Kaggle.\
- ✅ Toujours accessible via une URL publique.\
- ✅ Facile à partager.

**Inconvénients :** - ⚠️ Taille limitée (\<100 Mo).\
- ⚠️ Nécessite une mise à jour manuelle si le dataset change.

------------------------------------------------------------------------

# ✅ Conclusion

-   Pour un test rapide → `sample_data`.\
-   Pour conserver vos fichiers → Google Drive.\
-   Pour automatiser et utiliser beaucoup de datasets → API Kaggle.\
-   Pour partager et éviter les problèmes de connexion → GitHub.
