# Méthode `predire()`

## Objectif
La méthode `predire()` est utilisée pour générer des prédictions à partir du modèle entraîné.  
Elle applique le modèle sur l’ensemble de test (X_test) et produit les valeurs prédites des frais d’assurance.

---

## Étapes principales
1. Vérifier que le modèle a bien été entraîné (`self.modele` non nul).
2. Appliquer la fonction `.predict()` de Scikit-Learn sur les données `X_test`.
3. Sauvegarder les résultats dans un attribut de la classe (`self.y_pred`) pour analyse ultérieure.

---

## Détails techniques
- **Entrée** : `self.X_test` (données de test après prétraitement).
- **Sortie** : `self.y_pred` (vecteur numpy des valeurs prédites).

```python
def predire(self):
    if self.modele is None:
        raise ValueError("Le modèle n'a pas encore été entraîné.")
    self.y_pred = self.modele.predict(self.X_test)
    print("Prédictions terminées. Nombre de prédictions :", len(self.y_pred))
    return self.y_pred
```

---

## Exemple d’utilisation
```python
df_predictor.entrainer_modele(degre=1)
y_pred = df_predictor.predire()
print(y_pred[:10])  # Afficher les 10 premières prédictions
```

---

## Remarques
- Les prédictions sont basées sur le modèle choisi (linéaire ou polynomiale).
- Elles seront évaluées ensuite avec la méthode `evaluer()` pour mesurer leur précision.
