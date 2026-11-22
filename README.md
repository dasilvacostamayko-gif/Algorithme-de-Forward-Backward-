# Optimisation convexe en traitement d'images — ISTA & FISTA

Ce projet universitaire porte sur l'étude et l'implémentation de **méthodes d'optimisation convexe non différentiable**, avec des applications aux problèmes inverses et au traitement d'images.

L'objectif est d'étudier les fondements mathématiques des algorithmes **ISTA (Iterative Shrinkage-Thresholding Algorithm)** et **FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)**, puis de comparer leurs performances numériques.

## Applications étudiées

- **LASSO :** reconstruction de signaux parcimonieux à partir de mesures bruitées, avec une comparaison des vitesses de convergence d'ISTA et FISTA.
- **Variation totale (modèle ROF) :** débruitage d'images affectées par un bruit gaussien, à l'aide d'une formulation duale et d'un algorithme de gradient projeté.

## Aspects mathématiques

Le projet aborde notamment l'analyse convexe, les opérateurs proximaux, les sous-différentiels, la dualité convexe et l'étude des taux de convergence des algorithmes d'optimisation.

## Technologies utilisées

Python, NumPy, Matplotlib, Jupyter Notebook et LaTeX.

Le dépôt contient les implémentations Python, les expériences numériques ainsi qu'un rapport détaillant les formulations mathématiques et les résultats obtenus.
