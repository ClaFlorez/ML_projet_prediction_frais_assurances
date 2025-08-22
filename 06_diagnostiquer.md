# Fonction `diagnostiquer()`

## Description
Cette méthode permet d’évaluer et de diagnostiquer le modèle après l’entraînement.  
Elle calcule les métriques principales et affiche des graphiques de comparaison.

## Étapes détaillées
1. **Calcul des prédictions**  
   - Génération des prédictions sur `self.X_test`.  

2. **Évaluation quantitative**  
   - Calcul du **MSE (Mean Squared Error)**.  
   - Calcul du **R² (coefficient de détermination)**.  

3. **Visualisation**  
   - Graphique `y_test` (réels) vs `y_pred` (prédictions).  
   - Permet d’identifier les écarts du modèle.

## Résultat attendu
- Mesures chiffrées de performance affichées.  
- Visualisation des prédictions vs valeurs réelles.
