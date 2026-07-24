# Finsler-MDS : extensions géodésiques et données asymétriques

Ce dépôt rassemble le travail réalisé au cours d'un stage de recherche sur
l'apprentissage de variété pour des données asymétriques. Il prolonge
[Finsler Multi-Dimensional Scaling](https://arxiv.org/abs/2503.18010)
([code d'origine](https://github.com/Tommoo/FinslerMDS)), qui étend le MDS
classique à des matrices de dissimilarités non symétriques en utilisant une
métrique de Randers dans l'espace d'arrivée.

Le dépôt se concentre sur trois prolongements :

- des métriques de Finsler plus riches, en particulier la **métrique de
  Matsumoto** ;
- **Finsler-GeoMDS**, qui compare les dissimilarités cibles à des distances
  géodésiques dans l'embedding plutôt qu'aux seules distances directes ;
- l'application de ces méthodes à des données synthétiques et à deux jeux de
  données de biologie unicellulaire, Paul15 et Pancreas.

Il s'agit de code de recherche. Les expériences biologiques et plusieurs
optimiseurs alternatifs sont exploratoires : le dépôt sert à reproduire les
essais du stage et à en lancer de nouveaux.

## Méthodes implémentées

### Finsler-MDS

Le Finsler-MDS direct minimise un stress entre une matrice de dissimilarités
cibles, éventuellement asymétrique, et les distances de Finsler mesurées sur
les segments reliant les points de l'embedding.

Deux optimiseurs principaux sont disponibles :

- `smacof_randers` est l'algorithme Finsler-SMACOF spécialisé pour Randers.
  L'implémentation par défaut corrige la majoration et la mise à jour de la
  version publiée ;
- `gradient_descent` minimise directement le stress et accepte toutes les
  métriques implémentées.

### Finsler-GeoMDS

Finsler-GeoMDS remplace les distances directes dans l'espace d'arrivée par des
plus courts chemins sur un graphe kNN reconstruit à partir de l'embedding.
L'objectif peut ainsi préserver une géométrie de chemins qu'un MDS direct
écraserait ou déformerait.

L'optimiseur principal est `path_frozen`. Il alterne entre :

1. la construction du graphe courant et le calcul de ses plus courts chemins ;
2. quelques pas d'optimisation pendant lesquels ces chemins sont gardés fixes.

Des options de *landmarks*, de sous-échantillonnage des cibles et de contraintes
locales directes permettent de réduire le coût sur les jeux de données de
plusieurs milliers de points. Path-Frozen reste toutefois plus lent et plus
sensible à l'initialisation que les méthodes à distances directes.
Ces heuristiques et leurs principaux paramètres sont détaillés dans
[Réglage de Path-Frozen](#réglage-de-path-frozen).

Le dépôt contient aussi des approches différentiables de plus court chemin :
`datasp` (soft Floyd-Warshall), `soft_bellman_ford` et
`relaxed_bellman_ford`. Elles servent surtout de méthodes expérimentales et de
comparaison ; dans les essais, Path-Frozen offre le meilleur compromis
entre coût et qualité.

### Finsler-UMAP

`finsler_umap` est une variante asymétrique d'UMAP utilisant également une métrique de Finsler dans l'espace
d'arrivée. Il est tiré de [Harnessing Data Asymmetry](https://arxiv.org/abs/2603.11396) mais a été modifié
pour prendre en entrée une matrice de dissimilarité asymétrique quelconque, au lieu d'utiliser uniquement
des effets asymétriques venant de la densité non uniforme des points. Comme dans l'article, il n'est pas tout à fait équivalent à UMAP même dans le cas symétrique.

### Métriques

Les métriques de Finsler de l'espace d'arrivée sont définies dans
`finsler_mds/metrics.py`. Leur direction privilégiée est toujours le dernier
axe de l'embedding.

| Métrique | Usage principal |
| --- | --- |
| `RandersMetric` | métrique de Finsler la plus simple, utilisée dans le papier Finsler-MDS |
| `MatsumotoMetric` | dépendance non linéaire à la direction, peut être vue comme un temps de déplacement le long d'une pente ; |
| `ConvexifiedMatsumotoMetric` | version "corrigeant" la non-convexité de Matsumoto pour ‖ω‖ > 1/2 ; |
| `ToblerMetric`, `ConvexifiedToblerMetric`, `MinettiMetric` | modèles de déplacement sur pente implémentés pour des comparaisons exploratoires. |

Le sous-package `finsler_mds/evaluation` fournit le stress direct ou
géodésique, des mesures de préservation de l'asymétrie et les métriques
d'évaluation utilisées pour la RNA velocity.

## Installation

Depuis la racine du dépôt :

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` fixe l'environnement complet, y compris Scanpy, scVelo et
CellRank. Pour les seules expériences synthétiques, les dépendances essentielles
sont NumPy, SciPy, scikit-learn, Matplotlib, Joblib, Numba et `umap-learn`.

Certaines expériences demandent des dépendances supplémentaires :

- l'accélération GPU de certains optimiseurs requiert une version de CuPy
  adaptée à l'installation CUDA ;
- `main_paul15_monocle.py` requiert R, Monocle 3 et les paquets R indiqués
  dans `requirements.txt`.

Les jeux Paul15 et Pancreas sont chargés respectivement par Scanpy et CellRank.
Ils ne sont pas versionnés dans ce dépôt. Leur premier chargement et le calcul
de la RNA velocity peuvent demander du temps et un accès réseau.

## Utilisation de l'API

Le point d'entrée commun est `fit_finsler_mds`. Par exemple, pour optimiser un
Finsler-MDS direct avec Matsumoto :

```python
import numpy as np

from finsler_mds import MatsumotoMetric, fit_finsler_mds

D = np.array(
    [
        [0.0, 1.0, 2.0],
        [1.3, 0.0, 1.0],
        [2.4, 1.2, 0.0],
    ]
)

embedding, stress = fit_finsler_mds(
    D,
    metric=MatsumotoMetric(alpha=0.4),
    optimizer="gradient_descent",
    n_components=2,
    random_state=42,
)
```

Les principaux noms acceptés pour `optimizer` sont :

| Optimiseur | Objectif |
| --- | --- |
| `smacof_randers` | Finsler-MDS direct, Randers uniquement ; |
| `gradient_descent` | Finsler-MDS direct, métrique générique ; |
| `path_frozen` | Finsler-GeoMDS, méthode recommandée ; |
| `datasp` | Finsler-GeoMDS avec soft Floyd-Warshall ; |
| `soft_bellman_ford`, `relaxed_bellman_ford` | variantes géodésiques expérimentales ; |
| `finsler_umap` | graphe fuzzy orienté et objectif de type UMAP. |

Pour Path-Frozen, il est généralement préférable de partir d'un embedding déjà
raisonnable, par exemple obtenu avec UMAP, Isomap ou Finsler-MDS direct. Les
scripts donnent des configurations complètes pour chaque famille d'expérience.

## Détails sur Path-Frozen

### Fonctionnement et heuristiques

Comme dit plus tôt, Path-Frozen enchaîne des itérations externes qui consistent en un recalcul du graphe kNN des points de l'embedding, un calcul des plus courts chemins dans ce graphe en utilisant l'algorithme de Dijkstra et enfin quelques pas de descente de gradient (itérations internes) sur un stress où les chemins optimaux sont supposés rester optimaux (chemins "gelés").

Dans sa version complète, ceci est trop coûteux pour les graphes à partir de quelques milliers de points : exécution de Dijkstra depuis les n points à chaque itération externe, puis descente de gradient en prenant en compte les \(n(n-1)\) couples orientés.

Pour rendre le calcul
abordable, Path-Frozen utilise quelques heuristiques :

- n'effectuer Dijkstra que depuis un certain nombre de points, des **landmarks** ;
- pour chaque landmark, sous-échantillonner le nombre de cibles associées, pour réduire le nombre de paires prises en compte dans la descente de gradient ;
- pour couvrir la géométrie locale malgré ces heuristiques, ajouter au stress les *paires locales* (voisins selon les dissimilarités, c.-à-d. dans l'espace d'origine) en considérant leur distance directe et non géodésique pour éviter l'utilisation de Dijkstra.

Par ailleurs, pour limiter les problèmes liés à la sur-optimisation de l'objectif gelé durant la descente de gradient (les collisions entre branches notamment), nous avons ajouté :
- une régularisation optionnelle du stress utilisant les distances directes ;
- un amortissement des déplacements entre deux reconstructions du graphe.

### Paramètres importants

| Paramètre | Rôle et compromis |
| --- | --- |
| `graph_neighbors` | Nombre de voisins du graphe kNN construit dans l'embedding. Une valeur trop faible peut produire des chemins fragiles ou un graphe déconnecté ; une valeur trop élevée autorise davantage de raccourcis. |
| `outer_iter` | Nombre de reconstructions du graphe et de recalculs des plus courts chemins. |
| `inner_iter` | Nombre maximal d'itérations d'optimisation effectuées en gardant les chemins fixes. Une grande valeur permet des changements rapides, mais risque d'optimiser longtemps des chemins devenus obsolètes. |
| `outer_step_size` | Fraction du déplacement proposé qui est effectivement appliquée après une optimisation interne. `1` applique le déplacement complet ; une valeur plus faible amortit les changements de graphe et améliore généralement la stabilité, mais ralentit. |
| `n_local_pairs` | Nombre de plus petites dissimilarités conservées pour chaque point source, afin que les relations locales restent toujours représentées. Peut être pris égal à, ou proche de, graph_neighbors. |
| `local_pair_mode` | Avec `geodesic`, les couples locaux utilisent les chemins du graphe (à éviter, existe surtout pour mesurer la différence de temps et d'effet) ; avec `direct` (défaut), ils utilisent directement leurs distances dans l'espace d'embedding. |
| `n_landmark` | Nombre de points utilisés comme sources pour représenter la structure globale. Une valeur plus élevée donne un objectif plus complet, mais plus coûteux (10 % des points est une bonne base). |
| `random_landmark_fraction` | Proportion (entre `0` et `1`) des *landmarks* tirés aléatoirement, les autres étant choisis par *farthest-point sampling* dans les dissimilarités cibles. `0` privilégie une couverture fixe et régulière, `1` un échantillonnage entièrement aléatoire. |
| `resample_random_landmarks` | Si cette option est active (défaut), renouvelle la partie aléatoire des *landmarks* à chaque itération externe. La partie obtenue par *farthest-point sampling* reste fixe. N'est désactivé que pour des tests comparatifs, en général. |
| `targets_per_landmark` | Nombre maximal de cibles échantillonnées par *landmark* (les cibles étant resélectionnées aléatoirement à chaque itération externe). Les poids sont corrigés pour estimer l'objectif complet associé aux *landmarks*. Si la descente de gradient n'est pas le bottleneck du temps d'exécution, peut être pris assez haut pour plus de précision. Entre 20 et 50 % des points généralement. |
| `local_global_reweighting` | Contrôle l'équilibrage des groupes local et global : aucun avec `none`, par masse totale des poids avec `count` (défaut), ou par énergie cible (poids multipliés par `D^2` avant équilibrage, privilégie donc beaucoup le local) avec `energy`. |
| `local_weight` | Multiplie le poids du groupe local après cet éventuel rééquilibrage. Une valeur élevée favorise la structure locale. |
| `direct_stress_mode` | Formule pour la régularisation avec distance directe. Le mode `hinge` pénalise seulement les distances directes devenues trop petites ; le mode `mds` (défaut) ajoute un véritable stress MDS direct. |
| `direct_stress_weight` | Poids de la régularisation directe (sur tous les couples, non affectée par landmark/targets), désactivée avec `0`. Utile en début d'optimisation, mais doit être diminué pour ne pas trop biaiser le résultat. |
| `direct_stress_margin` | En mode `hinge`, fixe le seuil en dessous duquel une distance directe est pénalisée, comme fraction de la dissimilarité cible. |

### Optimisation en plusieurs phases

Il est généralement préférable de ne pas lancer Path-Frozen une seule fois
avec des paramètres fixes. On peut d'abord effectuer une phase exploratoire
assez agressive pour obtenir rapidement une bonne structure globale, puis
relancer Path-Frozen avec l'embedding obtenu comme nouvelle valeur de `init` et
des paramètres plus conservateurs pour restaurer la structure locale.

Une configuration exploratoire typique utilise par exemple
`inner_iter=50`, `outer_step_size=1`, `direct_stress_weight=0.3`,
`local_global_reweighting="count"` et `local_weight=0.1`. La phase de finition
peut ensuite utiliser `inner_iter=10`, `outer_step_size=0.2`, un
`direct_stress_weight` de `0.05`, `0.01` ou même `0`, et un `local_weight` de
`0.5` ou `1`. Les dernières phases peuvent encore réduire `inner_iter` et
`outer_step_size` si nécessaire.

Ces valeurs sont des points de départ et dépendent du jeu de données, de
l'initialisation et des autres heuristiques d'échantillonnage. Cet enchaînement
est actuellement réalisé explicitement dans certains scripts ; son automatisation
constituerait une amélioration utile de l'optimiseur.

## Expériences disponibles

Les scripts se lancent depuis la racine du dépôt avec
`python scripts/<nom_du_script>.py`. La plupart de leurs paramètres se trouvent
dans des constantes ou dictionnaires au début du fichier.

### Cas synthétiques

| Scripts | Expérience |
| --- | --- |
| `main_nested_rectangles_path_frozen.py` | cas contrôlé pour GeoMDS : deux rectangles imbriqués dont les distances géodésiques ne peuvent pas être bien représentées par un MDS direct ; |
| `main_parallel_bridges_path_frozen.py` | cas avec asymétrie que Finsler-GeoMDS peut mieux préserver que Finsler-MDS direct ; |
| `main_spiral_path_frozen.py` | cas synthétique montrant qu'on ne peut pas juste garder le graphe kNN d'origine dans Path-Frozen ; |
| `main_mountains.py` | géodésiques sur une surface à trois montagnes ; |
| `main_sea.py`, `main_sea_paths.py` | cartes de courants synthétiques, comparaison des métriques et visualisation de chemins source-cible. `sea_datasets.py` contient les générateurs associés ; |
| `main_branching.py` | dataset pour le benchmarking de temps d'exécution de Path-Frozen/soft-Bellman-Ford ; |
| `benchmark_branching_path_frozen.py`, `benchmark_soft_bellman_ford.py` | mesures de stress au cours du temps sur Branching et Swiss roll, pour comparer différents paramètres et heuristiques de Path-Frozen/soft-Bellman-Ford. |

Les figures (et dans certains cas les embeddings) sont écrites dans `scripts/res/`, qui est ignoré par Git.

### Exemples issus du papier Finsler-MDS

`main_swiss_roll_full.py`, `main_swiss_roll_hole.py` et `main_2D_maps.py`
conservent les expériences de visualisation du dépôt d'origine, avec quelques
adaptations pour utiliser l'API actuelle. Elles couvrent le Swiss roll, la
robustesse à un trou et les cartes de rivière ou de mer.

La partie Link Prediction est dans un autre dépôt : https://github.com/MorganMyr/FinslerLinkPrediction.

### Inférence de trajectoire sur Paul15

L'inférence de trajectoire donne à chaque point un pseudotime (avancement de la transformation biologique). On cherche à tester si MDS, GeoMDS ou leurs variantes Finsler (en utilisant le pseudotime voire la densité pour créer des dissimilarités asymétriques) donnent de meilleures visualisations. Le dataset actuel Paul15 manque de ground truth et aucune mesure quantitative n'a été faite pour l'instant.

- `main_paul15_finsler.py` est le point d'entrée principal. Il construit des
  dissimilarités géodésiques dans l'espace de diffusion, éventuellement rendues
  asymétriques par le pseudotime ou la densité, puis teste Finsler-MDS et
  Path-Frozen ;
- `main_paul15_baseline.py` construit les références Scanpy, PAGA, DPT et UMAP ;
- `main_paul15_diffmap_embedding.py` relance une configuration ciblée à partir
  des caches de diffusion ;
- `main_paul15_monocle.py` et le script R de `monocle3_bridge/` comparent
  Monocle 3 sur UMAP et sur un embedding GeoMDS ;
- `main_paul15_phate.py` compare MDS et GeoMDS comme méthodes de visualisation en fin de pipeline PHATE ;
- `main_paul15_pseudotime_lift.py` et
  `plot_paul15_paga_pseudotime_embeddings.py` produisent des visualisations
  complémentaires.

### RNA velocity sur Pancreas

Le pipeline Pancreas part de cellules avec chacune un vecteur de vélocité (direction d'évolution), tous projetés
dans un espace PCA. On interprète ces vecteurs comme des courants pour calculer des distances géodésiques asymétriques (selon une métrique de Randers ou Matsumoto) qui servent de dissimilarités. On peut ensuite appliquer l'une de nos méthodes (Finsler-MDS, GeoMDS, Finsler-UMAP, etc).

- `main_pancreas.py` contient le pipeline complet : chargement, prétraitement,
  RNA velocity, dissimilarités, initialisations, optimisation et sauvegarde ;
- `precompute_pancreas_velocity_distance_caches.py` recalcule rapidement
  plusieurs matrices de dissimilarités à partir d'un état biologique déjà mis
  en cache ;
- `evaluate_pancreas_embedding.py` évalue un embedding sauvegardé avec CBDir,
  la cohérence locale et globale des vélocités, la préservation des alignements
  et, si demandé, le stress ;
- `plot_pancreas_velocity_embedding.py` superpose le champ de vélocité projeté
  à un embedding ;
- `pancreas_gap_distance.py` évalue la gap distance, une métrique de sensibilité à l'absence d'une région, utilisée par exemple dans le papier VeloViz (ce script n'a que peu été utilisé, mais il est laissé si on veut poursuivre l'étude dans cette voie).

`finsler_mds/utils/pancreas_campaign.py` conserve des configurations et des
fonctions CSV réutilisables pour écrire des campagnes de tests sur Pancreas, plutôt que de les lancer individuellement via main_pancreas.

## Organisation du dépôt

```text
finsler_mds/
  api.py                 point d'entrée commun
  metrics.py             métriques de Finsler
  optimizers/            Finsler-MDS, GeoMDS et Finsler-UMAP
  evaluation/            stress, asymétrie et métriques RNA velocity
  utils/                 graphes, caches, initialisations et figures
scripts/                  expériences reproductibles
docs/                     notes et supports de travail
```

Les résultats, embeddings et caches sont normalement créés sous
`scripts/res/`. Ils ne sont pas suivis par Git. Les noms de fichiers et les métadonnées des embeddings encodent
une partie des paramètres utilisés pour les créer.

## Limites et perspectives

- L'implémentation Path-Frozen de Finsler-GeoMDS reste significativement plus lente qu'une gradient descent pour MDS normal. Elle pourrait être améliorée, ou une autre implémentation de GeoMDS pourrait être envisagée. 
- Path-Frozen nécessite pour l'instant souvent d'être lancé plusieurs fois à la suite, en réduisant le nombre inner_iter et outer_step_size par exemple. On pourrait automatiser ça.
- Il serait bien de développer un dataset synthétique où Finsler-MDS/GeoMDS/UMAP donnent clairement un meilleur embedding avec Matsumoto qu'avec Randers. Sur des données 3D figées, on peut montrer que Matsumoto a des géodésiques plus naturelles (contournement de montagnes), mais souvent Randers est aussi capable de donner un embedding satisfaisant.
- Nous n'avons pas encore de résultats quantitatifs en Trajectory Inference, qui est pourtant un bon domaine de test pour GeoMDS. Aborder un autre dataset que Paul15 qui aurait des ground truth de pseudotime connues serait intéressant.
- En RNA-velocity nous avons déjà des résultats acceptables sur Pancreas, mais tester d'autres métriques comme la Gap Distance ou essayer un autre dataset pourrait éventuellement donner des résultats plus favorables à Finsler-MDS (avec Matsumoto par exemple) ou GeoMDS.

## Référence principale

Si ce code est réutilisé, citer au minimum le papier Finsler-MDS dont il
prolonge l'implémentation :

```bibtex
@inproceedings{dages2025finsler,
  title     = {Finsler Multi-Dimensional Scaling: Manifold Learning for
               Asymmetric Dimensionality Reduction and Embedding},
  author    = {Dag{\`e}s, Thomas and Weber, Simon and Lin, Ya-Wei Eileen and
               Talmon, Ronen and Cremers, Daniel and Lindenbaum, Michael and
               Bruckstein, Alfred M. and Kimmel, Ron},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition},
  pages     = {25842--25853},
  year      = {2025}
}
```

Le code est distribué sous licence BSD 3-Clause ; voir `LICENSE`.
