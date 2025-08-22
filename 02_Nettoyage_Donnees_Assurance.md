# Fonction `nettoyer_donnees()`

## Description
Cette méthode réalise le nettoyage complet du dataset d’assurance.  
Elle applique plusieurs étapes de préparation afin de garantir la qualité et la cohérence des données.

## Étapes détaillées
1. **Conversion des types**  
   - Convertit les colonnes numériques (`age`, `bmi`, `charges`, `children`) en types adaptés (`int`, `float`).  
   - Gère les valeurs invalides en les transformant en `NaN` (via `errors="coerce"`).

2. **Gestion des valeurs manquantes**  
   - Affiche un résumé des valeurs manquantes.  
   - Supprime les lignes où des variables essentielles sont absentes : `age`, `smoker`, `bmi`, `charges`.  

3. **Suppression des doublons**  
   - Supprime les éventuelles lignes dupliquées.  
   - Affiche la quantité de doublons supprimés.

4. **Création d’une version propre**  
   - Après nettoyage, une copie est sauvegardée dans `self.df_clean`.  
   - Une seconde copie (`self.data_complet`) est conservée pour l’analyse future.

## Résultat attendu
- Dataset nettoyé, prêt pour l’exploration et l’entraînement des modèles.  
- Taille et structure du DataFrame mises à jour.

1.  Vérification des types de données.\
2.  Contrôle des valeurs manquantes et doublons.\
3.  Encodage One-Hot des variables catégorielles.\
4.  Standardisation des variables numériques (optionnel).\
5.  Séparation en **features (X)** et **target (y)**.
