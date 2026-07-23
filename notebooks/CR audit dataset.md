# Audit du dataset Dexed régénéré — compte rendu

Antonin Longeot — dataset complet (30 145 presets), migration DawDreamer + calcul TT/ACTM terminés.

## 1. Contexte

Migration du rendu audio (RenderMan → DawDreamer) terminée, dataset complet régénéré : 120 580 wav +
spectrogrammes. Calcul des descripteurs de timbre (TimbreToolbox + AudioCommons Timbral Models) terminé
sur les 30 145 presets, avec la sélection automatique par corrélation reproduisant exactement la méthode
du papier SpinVAE2 : **6 features ACTM + 32 features TT = 38 features** (`a ∈ R^38`), identique au papier.

## 2. Couverture de l'espace de comportement (Z, 38 features)

PCA : 43.6% de variance expliquée sur 2 composantes — confirme que l'espace est intrinsèquement
multidimensionnel, pas résumable par 2 axes.

![PCA de Z, coloré par ac_brightness](pca_z_brightness.png)

Le gradient de brillance est très propre et continu sur la PCA — la structure principale de l'espace
Z est bien capturée par les deux premières composantes.

UMAP (préserve la structure de voisinage local, contrairement à la PCA) :

![UMAP de Z, coloré par ac_brightness](umap_z_brightness.png)

Structure globalement connectée et richement texturée, avec une **excroissance nettement détachée**

(à droite sur la figure) — un groupe de presets très différent du reste, à investiguer (écoute de
quelques exemples à prévoir).

Comparaison de deux descripteurs (sémantique `ac_brightness` vs. spectral pur `tt_SpecCent_IQR`) sur
le même embedding :

![UMAP de Z, double coloration](umap_z_double_coloring.png)

Les deux features s'organisent spatialement de façon très cohérente, y compris sur le groupe détaché —
ce qui confirme que la structure observée n'est pas un artefact d'une seule feature.

**Couverture quantifiée (grille 50×50 sur l'embedding UMAP) : 671/2500 cellules occupées = 26.8%**,
distance moyenne au plus proche voisin = 0.0228.

## 3. Comparaison avec l'espace des paramètres bruts (Θ, 155 dimensions)

Même analyse effectuée directement sur les 155 paramètres DX7 bruts, pour comparer la structure de
l'espace de paramètres à celle de l'espace de comportement.

![PCA de Θ, coloré par ac_brightness](pca_theta.png)

Contraste marqué avec la PCA de Z : le gradient de brillance est ici **complètement mélangé**, sans
organisation spatiale visible — deux presets à paramètres proches peuvent sonner très différemment,
et inversement.

![UMAP de Θ, coloré par ac_brightness](umap_theta.png)

Structure elle aussi très différente de celle de Z : un noyau central dense entouré de nombreux petits
clusters isolés et de points épars, plutôt qu'un continuum connecté.

**Couverture Θ : 513/2500 cellules = 20.5%**, distance moyenne au plus proche voisin = 0.0141.

| | Z (comportement, 38D) | Θ (paramètres, 155D) |
|---|---|---|
| Occupation de grille (50×50) | 26.8% | 20.5% |
| Distance moy. au plus proche voisin | 0.0228 | 0.0141 |

La distance NN plus faible pour Θ malgré une occupation de grille plus faible s'explique probablement
par une structure bimodale (cœur très dense + périphérie éparse), alors que Z est plus uniformément
réparti. Ces différences de structure entre Θ et Z illustrent concrètement la non-linéarité du mapping
Θ→Z que le stage cherche à exploiter.

## 4. Prochaines étapes

- Investiguer le groupe de presets détaché sur la carte UMAP de Z (écoute, identification des UIDs).
- Quantifier la couverture directement dans l'espace à 38 dimensions (pas seulement sur l'embedding
  UMAP, qui n'est pas une isométrie).
- Premiers tests d'intégration adtool (exploration IMGEP) sur les paramètres Dexed bruts, en discussion.
