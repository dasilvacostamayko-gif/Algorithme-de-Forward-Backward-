
# Optimisation convexe en traitement d'images — ISTA & FISTA

Ce projet explore les **méthodes d'optimisation proximales** pour la résolution de problèmes d'optimisation convexe non différentiable, avec des applications à la reconstruction de signaux parcimonieux et au débruitage d'images.

L'objectif est d'étudier les fondements mathématiques de ces méthodes, de les implémenter en Python et d'analyser leur comportement numérique.

Le projet s'articule autour de deux algorithmes :

- **ISTA** (*Iterative Shrinkage-Thresholding Algorithm*)
- **FISTA** (*Fast Iterative Shrinkage-Thresholding Algorithm*)

Les applications étudiées concernent le problème LASSO et le modèle de Rudin–Osher–Fatemi (ROF) pour le débruitage d'images.

---

## 1. Cadre mathématique

Nous nous intéressons aux problèmes d'optimisation de la forme :

$$
\boxed{
\min_{x \in \mathbb{R}^n}
F(x)=f(x)+g(x)
}
$$

où :

- $f$ est une fonction convexe différentiable dont le gradient est $L$-Lipschitzien ;
- $g$ est une fonction convexe, propre et semi-continue inférieurement, dont l'opérateur proximal peut être calculé.

L'opérateur proximal de $g$ est défini par :

$$
\operatorname{prox}_{\gamma g}(x)
=
\underset{z\in\mathbb{R}^n}{\operatorname{argmin}}
\left\{
\frac12\|z-x\|_2^2+\gamma g(z)
\right\}.
$$

Ces méthodes combinent une étape de descente de gradient sur la partie différentiable et une étape proximale sur la partie non différentiable.

---

## 2. Algorithmes ISTA et FISTA

### ISTA — Iterative Shrinkage-Thresholding Algorithm

ISTA repose sur l'itération :

$$
x_{k+1}
=
\operatorname{prox}_{\gamma g}
\left(x_k-\gamma\nabla f(x_k)\right).
$$

Pour un pas constant $\gamma=1/L$, sous les hypothèses classiques de convergence, l'écart à la valeur optimale satisfait :

$$
F(x_k)-F^\star
=
\mathcal{O}\left(\frac{1}{k}\right).
$$

### FISTA — Fast Iterative Shrinkage-Thresholding Algorithm

FISTA introduit une étape d'extrapolation permettant d'accélérer la convergence :

$$
x_{k+1}
=
\operatorname{prox}_{\gamma g}
\left(y_k-\gamma\nabla f(y_k)\right).
$$

$$
t_{k+1}
=
\frac{1+\sqrt{1+4t_k^2}}{2}.
$$

$$
y_{k+1}
=
x_{k+1}
+
\frac{t_k-1}{t_{k+1}}
(x_{k+1}-x_k).
$$

Avec l'initialisation $y_0=x_0$ et $t_0=1$.

Cette accélération permet d'obtenir le taux théorique :

$$
F(x_k)-F^\star
=
\mathcal{O}\left(\frac{1}{k^2}\right).
$$

---

## 3. Application au problème LASSO

Le LASSO est un problème d'optimisation convexe permettant de reconstruire des signaux parcimonieux à partir de mesures linéaires.

On considère le problème :

$$
\boxed{
\min_{x\in\mathbb{R}^n}
\left\{
\frac12\|Ax-b\|_2^2
+\lambda\|x\|_1
\right\}
}
$$

où :

- $A\in\mathbb{R}^{m\times n}$ est la matrice d'observation ;
- $b\in\mathbb{R}^m$ représente les mesures bruitées ;
- $x\in\mathbb{R}^n$ est le signal recherché ;
- $\lambda>0$ est le paramètre de régularisation.

La régularisation $\ell_1$ favorise l'obtention de solutions parcimonieuses.

### Résolution par ISTA et FISTA

On pose :

$$
f(x)=\frac12\|Ax-b\|_2^2,
\qquad
g(x)=\lambda\|x\|_1.
$$

Le gradient de $f$ est :

$$
\nabla f(x)=A^\top(Ax-b).
$$

L'opérateur proximal de $g$ correspond au **seuillage doux** :

$$
\operatorname{prox}_{\gamma g}(x)_i
=
\operatorname{sign}(x_i)
\max\left(|x_i|-\gamma\lambda,0\right).
$$

Nous utilisons les algorithmes ISTA et FISTA afin de comparer leurs vitesses de convergence.

### Expérience numérique

Le problème inverse est construit selon :

$$
b=Ax^\star+\varepsilon,
$$

où $x^\star$ est un signal parcimonieux et $\varepsilon$ un bruit gaussien.

Les paramètres utilisés sont :

| Paramètre | Valeur |
|---|---|
| Dimension du signal | $n=500$ |
| Nombre de mesures | $m=63$ |
| Coefficients non nuls | $s=10$ |
| Écart-type du bruit | $\sigma=0.01$ |
| Régularisation | $\lambda=1$ |
| Nombre d'itérations | $N=1000$ |
| Pas | $\gamma=1/\|A\|_2^2$ |

Les résultats numériques mettent en évidence une diminution plus rapide de la fonction objectif avec FISTA qu'avec ISTA.

---

## 4. Débruitage d'images par variation totale — Modèle ROF

Nous étudions également le modèle de Rudin–Osher–Fatemi (ROF), utilisé pour le débruitage d'images.

Le problème d'optimisation est :

$$
\boxed{
\min_u
\left\{
\frac12\|u-b\|_2^2
+\lambda\operatorname{TV}(u)
\right\}
}
$$

où $b$ représente l'image bruitée et $u$ l'image reconstruite.

La variation totale isotropique discrète est définie par :

$$
\operatorname{TV}(u)
=
\sum_{i,j}
\sqrt{
(D_xu)_{i,j}^2+
(D_yu)_{i,j}^2
}.
$$

Elle permet de régulariser l'image tout en favorisant la préservation des discontinuités, notamment des contours.

### Formulation duale

La résolution du modèle ROF repose sur une formulation duale du problème.

En notant $D$ l'opérateur de gradient discret et $D^\ast$ son adjoint, on considère :

$$
\min_{p\in C}
\frac12\|b-\lambda D^\ast p\|_2^2,
$$

avec :

$$
C=
\left\{
p:\|p_{i,j}\|_2\leq1
\quad\forall(i,j)
\right\}.
$$

La résolution numérique utilise une méthode de gradient projeté sur l'ensemble convexe $C$.

La projection s'effectue par :

$$
(P_C(p))_{i,j}
=
\frac{p_{i,j}}
{\max(1,\|p_{i,j}\|_2)}.
$$

### Expérience numérique

Une image en niveaux de gris est dégradée par un bruit gaussien additif centré.

Les paramètres utilisés sont :

| Paramètre | Valeur |
|---|---|
| Bruit | Gaussien additif |
| Écart-type | $\sigma=0.25$ |
| Régularisation | $\lambda=0.15$ |
| Nombre d'itérations | $N=100$ |
| Pas numérique | $\alpha=1/(8\lambda^2)$ |

Les résultats montrent une réduction partielle du bruit, tout en conservant les principales structures géométriques de l'image.

L'étude numérique porte également sur la stabilisation des fonctions objectifs au cours des itérations.

---

## 5. Notions mathématiques abordées

- Optimisation convexe et analyse non différentiable
- Sous-différentiels et conditions d'optimalité
- Opérateurs proximaux et seuillage doux
- Méthodes forward-backward
- Algorithmes ISTA et FISTA
- Analyse des taux de convergence
- Régularisation $\ell_1$ et parcimonie
- Problèmes inverses
- Variation totale et dualité convexe
- Débruitage et reconstruction d'images

---

## 6. Technologies utilisées

- **Python** : implémentation des algorithmes
- **NumPy** : calcul matriciel et opérations numériques
- **Matplotlib** : visualisation des résultats
- **Jupyter Notebook** : expérimentations numériques
- **LaTeX** : rédaction du rapport mathématique

---

## 7. Références bibliographiques

**[1]** Beck, A., & Teboulle, M. (2009). *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*. SIAM Journal on Imaging Sciences, 2(1), 183–202.

**[2]** Tibshirani, R. J. (2013). *The Lasso Problem and Uniqueness*. Electronic Journal of Statistics, 7, 1456–1490.

**[3]** Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

**[4]** Rudin, L. I., Osher, S., & Fatemi, E. (1992). *Nonlinear Total Variation Based Noise Removal Algorithms*. Physica D, 60(1–4), 259–268.

---

## 8. Rapport du projet

Le dépôt accompagne un rapport mathématique détaillant :

- Les fondements théoriques d'ISTA et FISTA
- Les calculs des gradients et opérateurs proximaux
- Les résultats de convergence
- L'application au problème LASSO
- La formulation duale du modèle ROF
- Les expériences numériques et leur interprétation

**Projet universitaire — Mathématiques appliquées, optimisation convexe et traitement d'images.**
