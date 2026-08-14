# Outil de visualisation Flowers (adtool) — mode d'emploi

Serveur web pour naviguer les decouvertes dans une carte 2D, avec les diagnostics de
l'investigation (silence, phase bootstrap/guidee) directement colorables sur les points,
et un panneau d'analyse pour comparer deux runs (distribution par parametre ou par
descripteur) sans quitter le navigateur.

---

## Lancer

Exemple directement copiable — remplacer le chemin du run par un de ceux listes plus bas :

```bash
cd ~/projects/spinvae2/adtool/dexed
LD_LIBRARY_PATH=/data/anasynth_nonbp/manaconda3/lib \
/data/anasynth_nonbp/longeot/envs/spinvae_2_18_env/bin/python3 \
  -m adtool.user_tools.visu.server \
  --discoveries runs/confirm/full/seed0/discoveries \
  --config_file configs/visu_analysis.json
```

`full` (config confirmee : logit\_mutation + carrier\_floor + niche\_curiosity\_bias +
filter\_degenerate\_parents + algorithm\_mutation\_prob) est desormais le run de reference
a regarder en premier : c'est la seule config qui bat le random sur Vendi et couverture a
la fois (voir tableau plus bas). Remplacer `seed0` par `seed1`..`seed4` pour les 4 autres
graines, ou le chemin du run par un autre de ceux listes plus bas.

Puis ouvrir **http://127.0.0.1:8765/**

Ne pas ecrire le chemin entre chevrons : bash interprete `<...>` comme une redirection, et
la commande echoue avec `Aucun fichier ou dossier de ce type`.

- `--config_file` charge les analyses (highlights carte + panneau Analysis, voir plus
  bas). Sans lui, la carte s'affiche mais sans les filtres silence/bootstrap, et le
  panneau Analysis n'a pas de module a proposer.
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

## Les highlights de la carte 2D

Definis dans `helpers/discovery_highlights.py`, actives par `configs/visu_analysis.json`.
Ils reprennent les diagnostics du rapport, utilisables interactivement sur la carte :

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

## Le panneau Analysis — distributions par parametre / par descripteur

C'est la section avec **Random baseline** et **Analysis run** dans l'interface. Deux
outils independants.

### Random baseline (generer un pool aleatoire a la volee)

Sert seulement si tu veux generer un pool aleatoire directement depuis le navigateur au
lieu d'utiliser `runs/random/combined` qui existe deja. En general **pas necessaire** ici
— passe directement a "Analysis run" ci-dessous avec les runs existants.

| Champ | Valeur a mettre |
|---|---|
| Config path | `configs/dexed_imgep_nichebias.json` (n'importe laquelle de nos configs suffit : seules ses sections `system` et `explorer` sont utilisees, `parameter_map.sample()` est appele en boucle) |
| Iterations | `1000` |
| Seed | ce que tu veux, ex. `0` |

Le placeholder par defaut (`examples/grayscott/gray_scott.json`) est un exemple generique
d'adtool, pas fait pour Dexed — ne pas le laisser tel quel.

### Analysis run (comparer deux runs deja existants)

C'est la partie utile pour voir la distribution sur les parametres ou les descripteurs.

| Champ | Valeur a mettre |
|---|---|
| Config file | `configs/analysis_theta.json` pour les 144 parametres Dexed (Theta), ou `configs/analysis_z.json` pour les 38 descripteurs de comportement (Z) |
| Primary label | nom du run charge au demarrage du serveur (celui passe a `--discoveries`), ex. `full` |
| Comparison discoveries path | chemin d'un autre run a comparer, ex. `runs/random/combined/discoveries` |
| Comparison label | `random` |

Clique **Add Dataset** pour comparer contre plusieurs runs a la fois (ex. baseline +
niche-bias + random tous ensemble). Clique **Run analysis**, puis **Reload Analysis**
pour voir les figures : une courbe de densite par parametre/descripteur, une figure par
dimension (144 pour Theta, 38 pour Z), primary vs chaque comparaison superposees.

C'est l'equivalent interactif de `figures/theta_distribution.png` (deja genere), mais
parametre par parametre au lieu d'un resume agrege — utile pour repondre a "est-ce que CE
parametre precis bouge assez", comme sur `figures/theta_step_per_param.png`.

**Exemple concret pour voir l'effondrement dans Theta** (a lancer avec
`--discoveries runs/sweeps/bootstrap/bs100/discoveries`) :

| Champ | Valeur |
|---|---|
| Config file | `configs/analysis_theta.json` |
| Primary label | `baseline` |
| Comparison discoveries path | `runs/random/combined/discoveries` |
| Comparison label | `random` |

**Exemple concret pour voir le correctif fonctionner** (a lancer avec
`--discoveries runs/confirm/full/seed0/discoveries`, la commande par defaut ci-dessus) :

| Champ | Valeur |
|---|---|
| Config file | `configs/analysis_theta.json` |
| Primary label | `full` |
| Comparison discoveries path | `runs/random/combined/discoveries` |
| Comparison label | `random` |

Clique **Add Dataset** une seconde fois avec `runs/sweeps/bootstrap/bs100/discoveries` /
label `baseline` pour voir les trois superposees : `full` colle a `random` sur (presque)
tous les parametres, la `baseline` reste resserree. Correspond a
`figures/theta_distribution_full.png` (deja genere, chiffres dans le rapport,
Sec.~Root Cause: Carrier-Level Regression).

---

## Runs interessants a regarder

| Run | Chemin | Pourquoi |
|---|---|---|
| **full (base a regarder)** | `runs/confirm/full/seed{0..4}` | **le correctif complet** : seule config qui bat le random sur Vendi ET couverture (17.34 vs 15.80, 280.6 vs 136.4 cellules, 5 seeds) ; silence guide 6.8% contre 73.2% pour la baseline |
| baseline | `runs/sweeps/bootstrap/bs100` (+`_seed1`..`_seed4`) | la reference, effondrement net apres le bootstrap |
| logit + carrier seul | `runs/confirm/logit_carrier/seed{0..4}` | les deux fixes de GENERATION seuls, sans le filtrage cote selection -- silence guide 21.2%, insuffisant seul pour battre le random sur Vendi |
| filter seul | `runs/filter/filter_only/seed{0..4}` | filtrage des parents degeneres seul, sans logit/carrier_floor -- recupere deja l'essentiel du Vendi (15.28) et de la couverture (270.2) |
| filter + niche | `runs/filter/filter_niche/seed{0..4}` | filtrage + biais niche-curiosity, sans les fixes de generation |
| niche + algomut | `runs/nichebias/k5_algomut/seed0` | le premier levier qui ait marche seul (88% de la couverture du random), avant `full` |
| combo + silence_sink | `runs/nichebias_silencesink/seed0` | interaction negative -- couverture en baisse malgre les deux fixes |
| grand bootstrap | `runs/bigboot/bs2000_seed0` | 2000 aleatoires puis 500 guides : la bascule est spectaculaire avec `is_bootstrap` |
| sous-espace restreint | `runs/restricted/L1_core14/seed0` | seulement 14 parametres FM explores |
| random (reference) | `runs/random/combined` | a utiliser comme `Comparison discoveries path` dans a peu pres toutes les analyses |

Pour le probleme d'origine, le plus parlant d'un coup d'oeil reste **`bigboot`** avec le
filtre `is_bootstrap`, ou l'on voit les 2000 points aleatoires couvrir largement l'espace
et les 500 points guides se tasser dans un coin. Pour le correctif, **`full`** avec le
filtre `is_silent` : contrairement a la baseline, les points guides ne s'accumulent plus
sur l'attracteur de silence.

---

## Regenerer les analyses hors ligne (sans passer par le navigateur)

```bash
cd ~/projects/spinvae2/adtool/dexed
LD_LIBRARY_PATH=/data/anasynth_nonbp/manaconda3/lib \
/data/anasynth_nonbp/longeot/envs/spinvae_2_18_env/bin/python3 scripts/coverage_curve_v3.py
```

Les figures atterrissent dans `figures/`. Le detail complet des resultats (Vendi,
couverture, silence) est aussi dans `imgep_investigation.ipynb`, sections 19-20.
