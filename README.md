
# Projet Big Data : Clustering avec k-Means

## Description
Ce projet est une partie du cours de Big Data à l'ESIREM, dirigé par Sergey Kirgizov. L'objectif est de se familiariser avec les techniques de clustering, en particulier l'implémentation de l'algorithme k-Means, et les techniques de visualisation de données.

## Technologies Utilisées
- Python
- NumPy
- Scikit-learn (sklearn)
- Matplotlib

## Installation
Pour exécuter ce projet, clonez le projet et assurez-vous que Python est installé sur votre système, ainsi que les bibliothèques susmentionnées. Vous pouvez les installer via pip :
```bash
pip install -r requirements.txt
```

## Structure du Projet

### 1. Génération de Données Artificielles
Le script génère un ensemble de données artificielles à l'aide de la fonction `make_blobs` de sklearn pour tester l'algorithme k-Means. Les paramètres tels que le nombre de centres et le nombre d'échantillons sont configurables.

### 2. Implémentation de l'Algorithme k-Means
Le projet implémente l'algorithme de base de Lloyd pour le clustering k-Means. Les étapes clés comprennent :
- Initialisation des centres de clusters.
- Attribution de chaque point au centre le plus proche.
- Mise à jour des centres des clusters.
- Calcul et visualisation de l'erreur SSE (Sum of Squared Errors) à travers les itérations.

### 3. k-Means++
Une amélioration de l'algorithme de base, k-Means++, est également implémentée. Elle diffère dans la manière dont les centres initiaux sont choisis, visant à améliorer la vitesse de convergence et la qualité de la solution finale.

### 4. Mini-batch k-Means
Le projet inclut également une implémentation de l'algorithme mini-batch k-Means, adapté pour de grands ensembles de données. Ce module teste l'algorithme sur un échantillon plus large (10 000+ points).

### 5. Visualisation des Résultats
Des fonctions de visualisation sont fournies pour afficher les clusters et l'évolution de l'erreur SSE.

## Utilisation
Il faut d'abord installer les dépendances
Pour exécuter l'algorithme k-Means sur les données générées :
```python
kmeans(all_points, nb_centers, nb_iterations: optional)
```

Pour k-Means++ :
```python
kmeanspp(all_points, nb_centers, nb_iterations: optional)
```

Pour le mini-batch k-Means :
```python
kmeans_mini_batch(all_points, nb_centers, batch_size, nb_iterations)
```

Le nombre de clusters ainsi que le nombre total de points peuvent être modifiés en modifiant les variables **nb_centers** et **num_samples**.

