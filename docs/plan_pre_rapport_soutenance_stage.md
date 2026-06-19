---
title: "Plan propose pour le pre-rapport et la soutenance anticipee"
subtitle: "Stage de recherche : extensions de Finsler-MDS et applications a des donnees dynamiques"
author: "Document de travail"
date: "18 juin 2026"
lang: fr
geometry: margin=2.3cm
papersize: a4
fontsize: 10.5pt
toc: true
numbersections: true
---

\newpage

# Positionnement general

## Message scientifique a porter

Le stage peut etre presente comme un travail d'approfondissement d'un cadre
recent de reduction de dimension pour donnees asymetriques : Finsler-MDS. Le
point central n'est pas simplement "appliquer Finsler-MDS a la biologie", mais
plutot :

> Comment etendre Finsler-MDS pour mieux representer des geometries dirigees,
> et comment evaluer ces extensions sur des donnees ou l'orientation a un sens
> scientifique, comme RNA velocity ou trajectory inference ?

Cette formulation est honnete par rapport au deroulement du stage : la methode
de depart existait deja, mais le travail effectue consiste a explorer ses
extensions naturelles, a comprendre leurs proprietes, a developper des
algorithmes utilisables, puis a chercher des validations experimentales.

## Contributions a revendiquer clairement

Il faut distinguer trois niveaux de contribution. Cela evitera que le rapport
ressemble a une suite d'essais experimentaux.

1. **Contribution theorique et geometrique.**  
   Etude de metriques alternatives a Randers dans le cadre canonique de
   Finsler-MDS, en particulier la metrique de Matsumoto et sa version
   convexifiee. Cette partie peut inclure les conditions de validite, les
   gradients, l'interpretation geometrique, et les effets attendus sur les
   directions favorisees.

2. **Contribution algorithmique.**  
   Developpement et comparaison de plusieurs optimiseurs pour des embeddings
   Finsler : gradient descent, version corrigee de Finsler-SMACOF, Finsler-UMAP,
   et surtout extensions geodesiques avec `path-frozen` et `soft-BF`. Les
   heuristiques de passage a l'echelle, en particulier landmarks, limitation des
   targets et paires locales directes, doivent etre presentees comme une
   contribution technique importante.

3. **Contribution experimentale et applicative.**  
   Construction d'un protocole d'evaluation sur donnees synthetiques et sur RNA
   velocity, avec comparaison a UMAP, Isomap/MDS et variantes. Les resultats
   actuels sont partiels mais exploitables : certains montrent que les variantes
   Finsler ameliorent des metriques d'orientation, tandis que d'autres revelent
   des limites importantes, notamment pour la geometrie globale de Pancreas.

## Angle a eviter

Ne pas presenter le stage comme : "j'ai recode beaucoup d'optimiseurs, puis
teste beaucoup de parametres". Il faut plutot raconter :

1. une question scientifique,
2. des hypotheses geometriques,
3. des methodes derivees de ces hypotheses,
4. des validations qui confirment ou infirment partiellement ces hypotheses.

Les grilles de parametres et les details de code doivent aller en annexe ou
etre resumes par quelques tableaux/figures.

\newpage

# Plan propose pour le pre-rapport

Le pre-rapport peut suivre la structure attendue d'un article de recherche. Le
fait que la soutenance soit anticipee doit etre mentionne dans l'introduction ou
dans une courte note liminaire : certains resultats sont finalises, d'autres
sont en cours et seront consolides pendant le dernier mois.

## Resume et note liminaire

**Objectif.** Donner en 10-15 lignes la question, les contributions et l'etat
d'avancement.

Contenu recommande :

- Finsler-MDS est un cadre recent pour representer des dissimilarites
  asymetriques.
- Le stage etudie deux extensions : changer la metrique de Finsler utilisee
  dans l'espace d'arrivee et remplacer les distances directes par des distances
  geodesiques sur un graphe.
- Ces extensions sont implementees et testees sur des donnees synthetiques et
  biologiques.
- Les resultats actuels montrent des gains sur certaines metriques
  d'orientation, mais aussi des limites sur la preservation globale de
  trajectoires complexes.
- Le dernier mois du stage doit consolider les comparaisons et les validations
  applicatives.

## 1. Introduction

### 1.1 Contexte scientifique

Expliquer le probleme general : beaucoup de donnees contiennent une information
directionnelle ou asymetrique. Exemples :

- graphes diriges ;
- distances de deplacement dans un champ de courant ;
- donnees biologiques dynamiques, ou les cellules sont associees a une direction
  d'evolution estimee ;
- pseudotime et trajectoires cellulaires.

Message cle : une visualisation symetrique peut separer ou rapprocher les bons
points, mais elle ne sait pas directement encoder le fait que "aller de \(i\)
vers \(j\)" n'a pas le meme cout que "aller de \(j\) vers \(i\)".

### 1.2 Probleme precis

Formuler explicitement :

> Etant donne une matrice de dissimilarites dirigees \(D_{ij}\), ou un graphe
> local dirige derive de donnees dynamiques, comment construire un embedding de
> faible dimension qui preserve a la fois la structure geometrique et les
> orientations pertinentes ?

Ce probleme est non trivial pour plusieurs raisons :

- les distances asymetriques ne peuvent pas etre preservees par une metrique
  euclidienne standard ;
- les methodes type UMAP preservent surtout des voisinages locaux et peuvent
  deformer les distances globales ;
- les distances geodesiques dependent de chemins optimaux qui changent pendant
  l'optimisation ;
- dans RNA velocity, les metriques standards evaluent souvent toute la pipeline
  "calcul des vitesses + embedding + projection", alors que le stage fixe les
  vitesses et etudie surtout l'etape d'embedding.

### 1.3 Objectif du stage

Objectif a annoncer :

> Etendre et evaluer Finsler-MDS comme outil de visualisation et de reduction de
> dimension pour donnees asymetriques, avec un accent sur les metriques de
> Finsler alternatives, les distances geodesiques, et les applications a des
> trajectoires biologiques.

### 1.4 Contributions annoncees

Proposer une liste courte :

- analyse de metriques de type \(\alpha\)-\(\beta\), notamment Matsumoto ;
- implementation d'optimiseurs directs et geodesiques ;
- heuristiques de passage a l'echelle pour `path-frozen` ;
- adaptation de Finsler-UMAP au cas de dissimilarites asymetriques externes ;
- protocole d'evaluation sur donnees synthetiques et RNA velocity ;
- resultats experimentaux et analyse des limites.

## 2. Etat de l'art

L'etat de l'art doit etre organise par familles, pas article par article.

### 2.1 Reduction de dimension par preservation de distances

Articles et themes a mobiliser :

- MDS classique et stress metric ;
- SMACOF comme algorithme majeur pour MDS ;
- Isomap comme MDS applique a des distances geodesiques ;
- limites des approches euclidiennes pour les dissimilarites asymetriques.

Positionnement :

- Le stage garde l'idee de stress MDS, mais remplace la metrique euclidienne de
  l'espace d'arrivee par une metrique de Finsler.
- La partie geodesique du stage se rapproche d'Isomap, mais optimise
  explicitement l'embedding au lieu de calculer une fois pour toutes des
  distances geodesiques dans l'espace source.

### 2.2 Methodes par graphes et voisinages locaux

Themes :

- UMAP, t-SNE, PHATE eventuellement ;
- graphes kNN, fuzzy simplicial set, negative sampling ;
- compromis entre preservation locale, structure globale et scalabilite.

Positionnement :

- Finsler-UMAP est une tentative d'utiliser la logique UMAP avec une geometrie
  asymetrique dans l'espace d'arrivee.
- Les tests montrent que ce cadre est prometteur mais sensible aux choix de
  support, de densite locale et de symetrisation.

### 2.3 Embeddings asymetriques et Finsler-MDS

Articles centraux :

- papier original Finsler-MDS ;
- follow-up "Harnessing Asymmetric Data" ;
- eventuellement travaux sur graphes diriges, embeddings hyperboliques ou
  embeddings asymetriques si utiles.

Points a expliquer :

- une metrique de Finsler peut verifier \(F(u) \neq F(-u)\) ;
- l'espace canonique de Randers encode une direction privilegiee simple ;
- l'asymetrie de \(D_{ij}\) peut etre traduite en organisation verticale de
  l'embedding.

Positionnement :

- Le stage approfondit ce cadre en etudiant d'autres metriques que Randers et
  en explorant des distances geodesiques dans l'espace d'arrivee.

### 2.4 Distances geodesiques et optimisation de chemins

Themes :

- shortest paths sur graphes ;
- distances geodesiques sur manifolds approximees par kNN ;
- relaxation differentiable type soft shortest path / soft Bellman-Ford ;
- cout computationnel des chemins et besoin d'heuristiques.

Positionnement :

- `path-frozen` est une approximation pragmatique : on gele les chemins optimaux
  pendant une sous-optimisation, puis on les recalcule.
- `soft-BF` est plus differentiable mais potentiellement plus couteux ou moins
  efficace.
- Les experiences stress-temps servent a justifier ces choix.

### 2.5 RNA velocity et trajectory inference

Themes :

- RNA velocity : estimation d'un champ de vitesses cellulaires ;
- scVelo comme baseline solide et largement utilisee ;
- visualisation habituelle via UMAP ;
- VeloViz et autres approches qui modifient la visualisation ou le graphe ;
- metriques : CBDir, ICVCoh, gap distance, trajectory consistency, coherence
  d'orientation.

Positionnement :

- Le stage n'essaie pas d'ameliorer le calcul des vitesses, mais l'etape
  d'embedding/projection.
- Les metriques de RNA velocity sont donc reutilisees dans un cadre legerement
  different : vitesse fixee, embedding variable.

## 3. Methodes

Cette section doit etre la plus precise du rapport. Elle peut etre divisee en
parties correspondant aux contributions.

### 3.1 Notation commune

Introduire :

- donnees \(x_i\) ;
- embedding \(y_i \in \mathbb{R}^d\) ;
- dissimilarites dirigees \(D_{ij}\) ;
- poids \(w_{ij}\) ;
- metrique de Finsler \(F_\theta(u)\), avec direction canonique le dernier axe.

Equation de base :

\[
\min_Y \sum_{i \ne j} w_{ij}
\left(F_\theta(y_j-y_i)-D_{ij}\right)^2.
\]

Insister sur le fait que \(F_\theta(y_j-y_i)\) peut etre different de
\(F_\theta(y_i-y_j)\).

### 3.2 Metriques de Finsler et alternatives a Randers

Presenter Randers :

\[
F_R(u) = \|u\| + \alpha u_d, \qquad 0 \leq \alpha < 1.
\]

Presenter Matsumoto :

\[
F_M(u) = \frac{\|u\|^2}{\|u\|-\alpha u_d}.
\]

Points a developper :

- interpretation directionnelle ;
- domaine de validite ;
- croissance plus forte que Randers dans certaines directions ;
- version convexifiee lorsque la boule unite n'est plus convexe ;
- lien avec les resultats theoriques obtenus pendant le stage.

Resultats a placer ici :

- profils \(\phi(s)\) des metriques ;
- schemas de boules unitaires ;
- consequences attendues sur les embeddings.

### 3.3 Construction des dissimilarites asymetriques

Decrire deux contextes.

**Donnees synthetiques.**  
Les dissimilarites proviennent d'une metrique connue ou d'un champ de courant.
Cela permet une validation qualitative et quantitative, car la solution attendue
est controlee.

**RNA velocity.**  
Les vitesses sont calculees avec scVelo, projetees en PCA, puis utilisees pour
modifier localement les dissimilarites entre voisins. Il faut expliquer que :

- les vitesses sont prises comme donnees ;
- l'embedding est l'objet de l'evaluation ;
- plusieurs formules angulaires ont ete testees, notamment une formule de type
  Randers et une formule exponentielle ;
- le clipping du cosinus controle l'amplitude de l'asymetrie.

### 3.4 Optimisation directe : SMACOF et gradient descent

Sous-parties conseillees :

- Finsler-SMACOF legacy et version corrigee ;
- role de `project_on_V` et limites avec poids non uniformes ;
- gradient descent/L-BFGS-B comme baseline robuste ;
- comparaison qualitative : SMACOF peut etre rapide mais sensible, gradient
  descent est plus flexible pour Randers, Matsumoto et metriques convexifiees.

Resultat important a mentionner :

- les versions corrigees de SMACOF doivent etre distinguees des versions legacy ;
- dans plusieurs tests Pancreas, gradient descent donne des resultats plus
  stables que SMACOF pour certaines metriques.

### 3.5 Finsler-MDS geodesique

Objectif :

\[
\min_Y \sum_{i \ne j} w_{ij}
\left(d^Y_{ij}-D_{ij}\right)^2,
\]

ou \(d^Y_{ij}\) est une distance de plus court chemin dans un graphe construit
sur l'embedding, avec longueurs d'aretes mesurees par \(F_\theta\).

Presenter les deux approches :

- `soft-BF` : relaxation differentiable de Bellman-Ford ;
- `path-frozen` : chemins optimaux geles pendant une etape d'optimisation.

Heuristiques a decrire :

- landmarks ;
- nombre limite de targets par landmark ;
- paires locales directes ;
- reweighting local/global ;
- choix random vs farthest-point sampling des landmarks ;
- suivi stress total vs temps hors cout d'evaluation.

Validation :

- Swiss roll : geometrie simple mais asymetrique ;
- branching dataset : geometrie geodesique plus difficile ;
- courbes stress-temps pour montrer l'interet des heuristiques.

### 3.6 Finsler-UMAP

Presenter la logique :

\[
p_{ij}
= \exp\left(
-\frac{\max(0,D_{ij}-\rho_{ij})}{\sqrt{\sigma_i\sigma_j}}
\right)
\]

ou une variante selon symetrisation de \(\rho\), \(\sigma\) et du support.
Contrairement a UMAP standard, on garde un graphe dirige, puis on optimise dans
un espace de Finsler.

Points importants :

- les \(p_{ij}\) ne sont pas des distances cibles comme en MDS ; ce sont des
  poids d'attraction ;
- \(p_{ij}>p_{ji}\) favorise une configuration ou \(d_F(i \to j)\) est plus
  petite que \(d_F(j \to i)\), donc une orientation dans l'espace canonique ;
- le negative sampling accelere l'optimisation et approxime la repulsion ;
- les choix de symetrisation sont critiques, car les densites locales peuvent
  introduire une asymetrie parasite.

Statut :

- implementation en cours de stabilisation ;
- resultats deja exploitables sur Pancreas ;
- comparaison a UMAP en cours.

### 3.7 One-step refinement

Presenter cela comme une strategie exploratoire mais tres lisible :

> partir d'un UMAP existant et appliquer un petit nombre d'etapes d'optimisation
> Finsler pour injecter une information directionnelle sans detruire la
> topologie locale de depart.

Pourquoi c'est interessant :

- UMAP est deja fort pour la coherence locale ;
- l'optimisation jusqu'au minimum du stress peut parfois degrader CBDir ou la
  lisibilite ;
- une correction faible peut agir comme regularisation orientee.

Ce qu'il faut valider :

- plusieurs seeds UMAP ;
- comparaison \(0, 1, 2, 3, 10, 100\) iterations ;
- interpolation entre UMAP et l'embedding apres une iteration ;
- metriques et distance a UMAP.

### 3.8 Metriques d'evaluation

Structurer en trois familles.

**Preservation de distances.**

- stress direct ou geodesique ;
- stress normalise ;
- stress vs temps.

**Qualite RNA velocity.**

- CBDir : coherence des transitions connues entre clusters ;
- ICVCoh : coherence locale des vitesses projetees ;
- gap distance : robustesse a un trou artificiel dans la trajectoire ;
- limites : ces metriques sont souvent utilisees pour evaluer toute une pipeline,
  alors qu'ici on fixe les vitesses.

**Metriques introduites ou adaptees pendant le stage.**

- Orientation Correlation : correlation de Spearman entre cosinus
  vitesse/voisin dans l'espace source et dans l'embedding ;
- Sign correctness : preservation du signe de ces cosinus ;
- GVCoh : coherence globale des vitesses projetees, par mean resultant length.

## 4. Resultats et analyse

Cette section doit separer les resultats finalises des resultats partiels.

### 4.1 Validation sur cas synthetiques controles

Figures possibles :

- cartes de courant `sea` ;
- chemins geodesiques dans un courant ;
- swiss roll asymetrique ;
- branching dataset.

Messages :

- Les cas simples verifient que les metriques de Finsler produisent bien une
  orientation dans l'embedding.
- Les cas geodesiques montrent pourquoi une distance directe entre points ne
  suffit pas toujours.

### 4.2 Efficacite des heuristiques de path-frozen

Resultats a montrer :

- courbes stress geodesique total vs temps sur branching et swiss roll ;
- comparaison all-pairs vs landmarks ;
- landmarks random vs farthest-point sampling ;
- effet des paires locales directes ;
- moyenne sur plusieurs seeds.

Interpretation attendue :

- all-pairs donne une reference mais est cher ;
- landmarks + targets limites reduisent fortement le cout ;
- les paires locales sont importantes pour ne pas stagner a un mauvais stress ;
- farthest-point landmarks peut ameliorer la couverture, surtout avec peu de
  landmarks, mais doit etre analyse par dataset.

### 4.3 Pancreas RNA velocity : baselines et Finsler-MDS

Baselines :

- UMAP 2D/3D ;
- Isomap/MDS ;
- eventuellement supervised UMAP comme comparaison cluster-aware, mais en
  precisant que ce n'est pas un baseline strictement comparable.

Resultats a presenter :

- tableau CBDir, ICVCoh, Orientation Correlation, Sign correctness, GVCoh ;
- visualisations UMAP vs meilleures variantes Finsler ;
- analyse par frontiere de CBDir, car les transitions Pre-endocrine vers Alpha,
  Beta, Delta, Epsilon ne sont pas egalement difficiles.

Message nuance :

- Certaines variantes Finsler ameliorent nettement la preservation locale de
  l'orientation.
- UMAP reste tres fort en ICVCoh et souvent en CBDir.
- Les optimisations plus poussees peuvent reduire le stress mais degrader la
  lisibilite ou CBDir.
- Les reweightings bases sur clusters/frontieres sont utiles pour comprendre les
  limites, mais moins "honnetes" comme methode principale.

### 4.4 One-step refinement sur Pancreas

Resultats a exploiter :

- campagne sur plusieurs seeds ;
- comparaison UMAP, interpolation, 1 iteration, 2 iterations, 10 iterations,
  100 iterations ;
- cas ou une seule iteration ameliore simultanement plusieurs metriques ;
- cas ou l'optimisation longue degrade CBDir malgre un stress meilleur.

Message :

- One-step refinement est une piste credible pour combiner les forces d'UMAP et
  de Finsler-MDS.
- Il doit etre presente comme une strategie de regularisation, pas comme la
  resolution complete du stress.

### 4.5 Finsler-UMAP sur Pancreas

Resultats actuels utilisables :

- En 2D, UMAP seed 42 donne environ  
  CBDir \(0.586\), ICVCoh \(0.843\), Orientation Corr. \(0.438\), Sign \(0.670\).
- Finsler-UMAP 2D atteint par exemple des compromis comme  
  `v1 a0.8` : CBDir \(0.572\), ICVCoh \(0.849\), Orientation Corr. \(0.554\),
  Sign \(0.731\).
- D'autres reglages ameliorent fortement l'orientation mais reduisent CBDir.

Interpretation :

- Finsler-UMAP peut injecter une information directionnelle tout en restant
  proche des performances locales d'UMAP.
- La methode reste sensible aux choix de symetrisation et au role des densites
  locales.
- Cette partie doit etre marquee comme "en cours" dans la soutenance anticipee.

### 4.6 Experiences en cours ou a finaliser

Les mentionner brievement, comme plan du dernier mois :

- comparaison Finsler-UMAP vs UMAP sur Pancreas et eventuellement Paul15 ;
- one-step refinement avec path-frozen et Finsler-UMAP ;
- comparaison fair path-frozen / soft-BF / DataSP ;
- trajectory inference avec pseudotime ground truth ;
- RNA velocity synthetique avec trajectoire connue ;
- trajectory consistency et gap distance si le protocole peut etre reproduit
  proprement.

## 5. Discussion

Organiser la discussion autour de questions, pas autour de scripts.

### 5.1 Ce qui fonctionne

- Les metriques de Finsler permettent bien d'encoder une direction privilegiee.
- Matsumoto donne parfois des comportements differents et utiles par rapport a
  Randers.
- `path-frozen` est un compromis pragmatique pour l'optimisation geodesique.
- Le one-step refinement est une piste simple et prometteuse.
- Finsler-UMAP peut ameliorer les metriques d'orientation.

### 5.2 Ce qui ne fonctionne pas encore completement

- Sur Pancreas, optimiser le stress n'augmente pas toujours CBDir.
- Les transitions multiples depuis Pre-endocrine creent une contrainte
  geometrique difficile.
- Les reweightings cluster-aware/frontier-aware peuvent ameliorer des scores,
  mais posent un probleme d'equite methodologique.
- Finsler-UMAP est sensible aux effets de densite locale.
- Les comparaisons de temps doivent etre faites avec prudence, car les objectifs
  et les couts par iteration different.

### 5.3 Limites du protocole experimental

- Une partie des resultats est encore dependante des hyperparametres.
- Pancreas est un dataset important, mais pas suffisant pour valider une methode
  generale.
- Les metriques RNA velocity ne sont pas parfaitement standard pour evaluer
  uniquement l'embedding.
- Il faut idealement ajouter un dataset synthetique avec ground truth connu.

## 6. Conclusion et perspectives

Conclusion possible :

- Le stage etend Finsler-MDS dans deux directions : nouvelles metriques et
  distances geodesiques.
- Il fournit une implementation experimentale unifiee et plusieurs protocoles
  d'evaluation.
- Les resultats montrent un potentiel pour les donnees dynamiques, surtout pour
  la preservation d'orientation, mais la validation applicative reste en cours.

Perspectives :

- finaliser Finsler-UMAP ;
- formaliser one-step refinement ;
- comparer rigoureusement les optimiseurs geodesiques ;
- valider sur trajectory inference avec pseudotime ;
- clarifier quelles metriques d'evaluation sont les plus pertinentes pour des
  embeddings de RNA velocity.

\newpage

# Plan propose pour la soutenance de 15 minutes

Objectif : environ 15 diapositives. Il faut eviter d'expliquer toutes les
campagnes. La soutenance doit montrer une histoire simple.

| Slide | Temps | Message | Visuel conseille |
|---|---:|---|---|
| 1 | 0:30 | Titre, contexte stage, Finsler-MDS | image simple d'embedding dirige |
| 2 | 1:00 | Probleme : visualiser des donnees asymetriques | schema \(D_{ij}\neq D_{ji}\) |
| 3 | 1:00 | Limite des embeddings euclidiens | exemple jouet ou courant |
| 4 | 1:15 | Idee Finsler-MDS | equation du stress + direction canonique |
| 5 | 1:15 | Contribution 1 : metriques alternatives | Randers vs Matsumoto, boules unitaires |
| 6 | 1:15 | Contribution 2 : distances geodesiques | schema path-frozen |
| 7 | 1:00 | Heuristiques de passage a l'echelle | landmarks + paires locales |
| 8 | 1:15 | Resultat synthetique/geodesique | stress-temps branching/swiss roll |
| 9 | 1:00 | Pourquoi RNA velocity ? | UMAP pancreas avec fleches |
| 10 | 1:15 | Metriques RNA velocity adaptees | CBDir, ICVCoh, orientation correlation |
| 11 | 1:30 | Resultats Pancreas : Finsler-MDS/GD | tableau court + visualisation |
| 12 | 1:15 | One-step refinement | UMAP, i1, i10/i100 |
| 13 | 1:15 | Finsler-UMAP : resultats partiels | compromis UMAP vs FUMAP |
| 14 | 1:00 | Limites et analyses negatives | pourquoi CBDir reste difficile |
| 15 | 1:00 | Conclusion et dernier mois | 3 contributions + next steps |

## Variante si le temps est trop court

Supprimer ou reduire :

- details de Finsler-UMAP ;
- details de cluster reweighting ;
- details de soft-BF/DataSP.

Garder absolument :

- probleme asymetrique ;
- Finsler-MDS et metriques alternatives ;
- path-frozen geodesique ;
- un resultat quantitatif stress-temps ;
- un resultat RNA velocity ;
- conclusion honnete sur les limites.

\newpage

# Figures et tableaux a preparer

## Figures prioritaires pour le rapport

1. **Schema general du pipeline.**  
   Donnees \(\to\) dissimilarites asymetriques \(\to\) choix de metrique
   Finsler \(\to\) optimiseur \(\to\) embedding \(\to\) evaluation.

2. **Metriques Randers/Matsumoto.**  
   Profils \(\phi(s)\), boules unitaires, interpretation directionnelle.

3. **Path-frozen.**  
   Schema : graphe kNN, plus courts chemins geles, sous-optimisation, mise a
   jour des chemins.

4. **Stress-temps des heuristiques.**  
   Branching et swiss roll, moyenne sur seeds, echelle log si necessaire.

5. **Pancreas : baselines.**  
   UMAP 2D/3D, Isomap/MDS, fleches de RNA velocity.

6. **Pancreas : meilleurs embeddings Finsler.**  
   Choisir peu de figures : un bon compromis, un cas ou l'optimisation longue
   degrade, un Finsler-UMAP.

7. **One-step refinement.**  
   Quatre images : UMAP, interpolation ou i1, i10, i100.

8. **Finsler-UMAP et densite.**  
   Figure montrant le role de \(\sigma_i\) ou des symetrisations si cette partie
   est gardee.

## Tableaux prioritaires

1. **Tableau des methodes.**  
   Methode, objectif, metrique possible, geodesique ou non, cout principal,
   statut.

2. **Tableau des metriques.**  
   CBDir, ICVCoh, Orientation Correlation, Sign correctness, GVCoh, stress,
   gap distance.

3. **Tableau Pancreas compact.**  
   UMAP vs meilleurs Finsler-MDS/GD/path-frozen/Finsler-UMAP, avec 4 ou 5
   metriques maximum.

4. **Tableau resultats partiels / prochains tests.**  
   A mettre plutot en conclusion ou annexe pour justifier la soutenance
   anticipee.

\newpage

# Comment presenter les resultats partiels

La soutenance anticipee peut etre geree proprement en utilisant trois etiquettes
dans le rapport ou oralement.

## Acquis

Elements que tu peux presenter comme contributions solides :

- implementation unifiee de plusieurs optimiseurs ;
- metriques Randers/Matsumoto/convexified Matsumoto ;
- correction et comparaison des variantes SMACOF ;
- path-frozen et heuristiques de passage a l'echelle ;
- pipeline d'evaluation Pancreas ;
- resultats one-step refinement sur plusieurs seeds ;
- premiers resultats Finsler-UMAP.

## En cours

Elements a presenter avec prudence :

- stabilisation de Finsler-UMAP ;
- interpretation fine des effets de densite et de symetrisation ;
- comparaison complete a UMAP/Isomap sur plusieurs datasets ;
- evaluation gap distance/trajectory consistency.

## Prevu pour le dernier mois

Priorite conseillee :

1. finaliser Finsler-UMAP sur Pancreas ;
2. formaliser one-step refinement avec une figure et un tableau robustes ;
3. produire une comparaison fair path-frozen vs soft-BF/DataSP ;
4. ajouter une validation trajectory inference avec pseudotime si possible ;
5. seulement ensuite explorer un autre dataset RNA velocity.

Cette priorisation est plus coherente que de multiplier les datasets sans
validation claire.

\newpage

# Conseils de redaction

## Ce qu'il faut mettre en avant

- Les questions de recherche, pas la chronologie.
- Les hypotheses geometriques : "telle metrique devrait favoriser telle
  orientation".
- Les validations negatives : elles montrent que tu as compris les limites.
- Les resultats quantitatifs, meme quand ils ne sont pas spectaculaires.
- Le fait que l'application biologique sert aussi de stress-test methodologique.

## Ce qu'il faut releguer en annexe

- longues grilles d'hyperparametres ;
- details de noms de fichiers ;
- resultats trop proches ou redondants ;
- essais avec reweighting de frontieres si leur interpretation est trop
  "tailor-made" ;
- scripts et details d'implementation bas niveau, sauf ceux qui justifient un
  gain algorithmique.

## Formulation utile pour l'introduction

> This internship investigates extensions of Finsler-MDS, a recent framework for
> embedding asymmetric dissimilarities. The work focuses on two complementary
> directions: enriching the geometry of the embedding space through alternative
> Finsler metrics, and replacing direct pairwise distances by graph geodesic
> distances. The resulting methods are evaluated on controlled synthetic
> manifolds and on biological dynamical data, where RNA velocity provides a
> natural source of directed information.

## Formulation utile pour les limites

> The goal of the internship is not to claim that Finsler embeddings universally
> outperform UMAP on RNA velocity datasets. Rather, the experiments identify
> regimes where Finsler geometry improves orientation preservation, and regimes
> where optimizing an asymmetric stress conflicts with commonly used visual
> quality metrics. This distinction is important for understanding where the
> method is promising and where further work is required.

\newpage

# Bibliographie a prevoir

La bibliographie finale devra etre precise, mais pour le plan il suffit de
preparer les familles suivantes.

## Reduction de dimension et MDS

- MDS classique et stress metric ;
- SMACOF ;
- Isomap ;
- UMAP ;
- eventuellement t-SNE/PHATE pour situer les baselines.

## Finsler et asymetrie

- Finsler-MDS original ;
- "Harnessing Asymmetric Data" ;
- travaux connexes sur embeddings de graphes diriges ou distances asymetriques.

## Optimisation de distances geodesiques

- shortest path differentiable ou relaxations soft shortest path ;
- methodes landmarks pour MDS/Isomap ;
- graph geodesic learning si pertinent.

## RNA velocity et trajectoires cellulaires

- RNA velocity original ;
- scVelo ;
- CellRank ;
- VeloViz ;
- benchmarking RNA velocity ;
- trajectory inference / pseudotime datasets si utilises.

\newpage

# Proposition de titre et d'accroche

Titres possibles :

1. **Extending Finsler-MDS for directed manifold learning**
2. **Finsler embeddings for asymmetric distances and biological trajectories**
3. **Beyond Randers Finsler-MDS: geodesic optimization and RNA velocity
   visualization**

Le troisieme titre est le plus precis, mais peut-etre trop technique. Pour une
soutenance de stage, le deuxieme est plus accessible.

Accroche orale possible :

> Classical embeddings preserve symmetric relations: two points are close or
> far. But in many datasets, especially dynamical biological data, the relation
> has a direction. My internship studies how Finsler geometry can encode this
> direction directly in the embedding space.

\newpage

# Version courte du fil narratif

1. Les donnees dynamiques induisent des relations asymetriques.
2. Les embeddings euclidiens standards ne peuvent pas representer directement
   cette asymetrie.
3. Finsler-MDS propose une solution recente avec une metrique de Randers.
4. Le stage etend ce cadre :
   - autres metriques, notamment Matsumoto ;
   - distances geodesiques avec path-frozen ;
   - adaptation UMAP et one-step refinement.
5. Les cas synthetiques valident les mecanismes attendus.
6. Les donnees RNA velocity montrent un potentiel pour preserver les
   orientations, mais aussi des limites importantes face a UMAP.
7. Le dernier mois sert a consolider les comparaisons et a ajouter une
   validation plus standard avec ground truth.

Ce fil narratif devrait etre la colonne vertebrale du pre-rapport et de la
soutenance.
