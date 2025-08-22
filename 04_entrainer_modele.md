# Fonction `entrainer_modele(degre=1)`

## Description
Cette méthode entraîne un modèle de régression sur les données préparées.  
Elle permet d’utiliser une régression linéaire simple ou polynomiale.

## Étapes détaillées
1. **Choix du degré**  
   - Si `degre=1` → Régression linéaire simple.  
   - Si `degre>1` → Transformation polynomiale des variables explicatives.

2. **Création du pipeline**  
   - Utilisation de `Pipeline` de Scikit-Learn.  
   - Étapes : transformation polynomiale (si nécessaire) → régression linéaire.

3. **Entraînement**  
   - Ajustement du modèle sur `self.X_train` et `self.y_train`.  
   - Sauvegarde du modèle dans `self.modele`.

## Résultat attendu
- Modèle de régression entraîné.  
- Prêt pour les prédictions.
