# Outil de visualisation Flowers (adtool) — mode d'emploi

Serveur web pour naviguer les decouvertes dans une carte 2D, avec les diagnostics de
l'investigation (silence, phase bootstrap/guidee) directement colorables sur les points.

---

## Lancer

Depuis `~/projects/spinvae2/adtool/dexed` :

```bash
LD_LIBRARY_PATH=/data/anasynth_nonbp/manaconda3/lib \
/data/anasynth_nonbp/longeot/envs/spinvae_2_18_env/bin/python3 \
  -m adtool.user_tools.visu.server \
  --discoveries runs/<CHEMIN_DU_RUN>/discoveries \
  --config_file configs/visu_analysis.json \
  --refresh
```

Puis ouvrir **<http://127.0.0.1:8765/>**

- `--config_file` charge les analyses (voir plus bas). Sans lui, la carte s'affiche mais
  sans les filtres silence/bootstrap.
- `--refresh` : mode live, le serveur surveille le dossier et ajoute les nouvelles
  decouvertes au fur et a mesure. A utiliser quand le run tourne encore. Donne aussi le
  bouton pause/reprise.
- Sans `--refresh` : mode manuel, chargement unique au demarrage (bouton `Refresh` pour
  recharger a la main).

Le demarrage prend **2 a 5 min** (chargement TensorFlow + calcul des filtres sur les
decouvertes existantes). C'est normal, le port ne repond pas avant.

### Acces depuis ta machine

Le serveur ecoute sur `127.0.0.1:8765` (local uniquement). En VSCode Remote-SSH le port
est generalement forwarde automatiquement (onglet *Ports*). Sinon, depuis ta machine :

```bash
ssh -L 8765:localhost:8765 bonsho
```

### Arreter

```bash
pkill -f adtool.user_tools.visu.server
```

---

## Les analyses disponibles

Definies dans `helpers/discovery_highlights.py`, activees par
`configs/visu_analysis.json`. Elles reprennent exactement les diagnostics du rapport,
mais utilisables interactivement sur la carte :

| Champ | Ce que c'est | A quoi ca sert |
|---|---|---|
| `peak_amplitude` | amplitude crete du rendu | reperer les sons faibles sans etre totalement silencieux |
| `is_silent` | rendu quasi silencieux (crete < 1e-4) | **voir l'attracteur de silence** : ou se concentrent les 54-67% de decouvertes mortes |
| `is_silence_sink` | point relocalise par le fix `silence_sink` | verifier que le fix disperse bien ces points hors de la zone normale |
| `is_bootstrap` | decouverte de la phase bootstrap (vs guidee) | **voir l'effondrement** : comparer visuellement ou atterrissent les deux phases |

Trois regles sont pre-configurees dans l'interface (`Quasi-silent discoveries` est active
par defaut). Le bouton **Compute Filters** materialise ces valeurs pour les decouvertes
qui n'en ont pas encore (utile en mode `--refresh` quand le run continue).

Chaque point peut etre affiche normalement, mis en evidence, ou masque selon la regle —
donc pour isoler visuellement la phase guidee, il suffit de masquer `is_bootstrap`.

---

## Runs interessants a regarder

| Run | Chemin | Pourquoi |
|---|---|---|
| baseline | `runs/sweeps/bootstrap/bs100` | la reference, effondrement net apres le bootstrap |
| niche + algomut | `runs/nichebias/k5_algomut/seed0` | le levier qui marche (88% de la couverture du random) |
| combo + silence_sink | `runs/nichebias_silencesink/seed0` | niche + relocalisation du silence |
| grand bootstrap | `runs/bigboot/bs2000_seed0` | 2000 aleatoires puis 500 guides : la bascule est spectaculaire avec `is_bootstrap` |
| sous-espace restreint | `runs/restricted/L1_core14/seed0` | seulement 14 parametres FM explores |

Le plus parlant pour comprendre le probleme d'un coup d'oeil : **`bigboot`** avec le
filtre `is_bootstrap`, ou l'on voit les 2000 points aleatoires couvrir largement l'espace
et les 500 points guides se tasser dans un coin.

---

## Distribution des descripteurs

La carte 2D est une projection (UMAP par defaut, changeable dans l'interface via
`Recompute Layout`). Pour la distribution des descripteurs bruts (les 38 dimensions de Z)
ou des parametres (Theta), c'est cote notebook :

- `imgep_investigation.ipynb` sections 19-20 : Vendi, couverture, silence
- `figures/theta_distribution.png` : distribution des 144 parametres explorables,
  genere par le script d'analyse Theta

---

## Regenerer les analyses hors ligne

```bash
cd ~/projects/spinvae2/adtool/dexed
LD_LIBRARY_PATH=/data/anasynth_nonbp/manaconda3/lib \
/data/anasynth_nonbp/longeot/envs/spinvae_2_18_env/bin/python3 scripts/coverage_curve_v3.py
```

Les figures atterrissent dans `figures/`.
