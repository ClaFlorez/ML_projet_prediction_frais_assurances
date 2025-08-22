# Fonction `pretraiter_donnees()`

## Description
Cette méthode prépare les données pour l’apprentissage automatique.  
Elle traite les variables catégorielles et divise le dataset en ensembles **train** et **test**.

## Étapes détaillées
1. **Encodage des variables catégorielles**  
   - Les colonnes `sex`, `smoker`, `region` sont transformées par **OneHotEncoder**.  
   - Cela permet de représenter les catégories sous forme binaire (0/1).

2. **Séparation des variables explicatives et cible**  
   - Variables explicatives (features) : toutes les colonnes sauf `charges`.  
   - Variable cible : `charges` (frais médicaux).

3. **Division train/test**  
   - Les données sont divisées en :  
     - 80% pour l’entraînement  
     - 20% pour le test  
   - Réalisé via `train_test_split` de Scikit-Learn avec une graine aléatoire fixe pour reproductibilité.

## Résultat attendu
- Données prêtes pour l’entraînement.  
- Attributs `self.X_train`, `self.X_test`, `self.y_train`, `self.y_test` créés.
