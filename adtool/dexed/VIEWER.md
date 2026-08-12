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
  --discoveries runs/bigboot/bs2000_seed0/discoveries \
  --config_file configs/visu_analysis.json
```

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
| Primary label | nom du run charge au demarrage du serveur (celui passe a `--discoveries`), ex. `bigboot` |
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

---

## Runs interessants a regarder

| Run | Chemin | Pourquoi |
|---|---|---|
| baseline | `runs/sweeps/bootstrap/bs100` | la reference, effondrement net apres le bootstrap |
| niche + algomut | `runs/nichebias/k5_algomut/seed0` | le levier qui marche (88% de la couverture du random) |
| combo + silence_sink | `runs/nichebias_silencesink/seed0` | interaction negative -- couverture en baisse malgre les deux fixes |
| grand bootstrap | `runs/bigboot/bs2000_seed0` | 2000 aleatoires puis 500 guides : la bascule est spectaculaire avec `is_bootstrap` |
| sous-espace restreint | `runs/restricted/L1_core14/seed0` | seulement 14 parametres FM explores |
| random (reference) | `runs/random/combined` | a utiliser comme `Comparison discoveries path` dans a peu pres toutes les analyses |

Le plus parlant pour comprendre le probleme d'un coup d'oeil : **`bigboot`** avec le
filtre `is_bootstrap`, ou l'on voit les 2000 points aleatoires couvrir largement l'espace
et les 500 points guides se tasser dans un coin.

---

## Regenerer les analyses hors ligne (sans passer par le navigateur)

```bash
cd ~/projects/spinvae2/adtool/dexed
LD_LIBRARY_PATH=/data/anasynth_nonbp/manaconda3/lib \
/data/anasynth_nonbp/longeot/envs/spinvae_2_18_env/bin/python3 scripts/coverage_curve_v3.py
```

Les figures atterrissent dans `figures/`. Le detail complet des resultats (Vendi,
couverture, silence) est aussi dans `imgep_investigation.ipynb`, sections 19-20.
