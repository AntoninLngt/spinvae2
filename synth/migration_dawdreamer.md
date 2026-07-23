# Migration du rendu audio Dexed : de RenderMan vers DawDreamer

## Contexte

SpinVAE2 utilise le synthétiseur Dexed (émulation du Yamaha DX7) pour le rendu audio
offline des presets. Le code original (`synth/dexed.py`) reposait sur **RenderMan**
(`librenderman`), une bibliothèque servant d'hôte VST. L'objectif était de faire tourner ce
pipeline sur les serveurs CPU du cluster anasynth (dotaku, bonsho, gottan).

## Problème

RenderMan ([`fedden/RenderMan`](https://github.com/fedden/RenderMan)) est un projet **conçu
pour Python 2 et non maintenu depuis environ 6 ans**. Sa compilation sur le cluster s'est
révélée impraticable : dépendances à l'API Python 2 disparue en Python 3, nécessité de
recompiler Boost localement sous fortes contraintes (pas de `sudo`, home NFS, environnements
conda partagés non modifiables), pour un résultat de toute façon instable. Poursuivre revenait
à maintenir indéfiniment du code mort ; une alternative moderne s'imposait.

## Solution retenue : DawDreamer

**DawDreamer** (David Braun) est l'équivalent moderne et maintenu de RenderMan, dont il est
l'évolution directe. Il s'installe en une commande (`pip install dawdreamer`), sans
compilation, et fonctionne nativement en Python 3.

> DawDreamer's foundation is [JUCE](https://github.com/juce-framework/JUCE), with a
> user-friendly Python interface thanks to [nanobind](https://github.com/wjakob/nanobind).
> DawDreamer evolved from an earlier VSTi audio "renderer",
> [RenderMan](https://github.com/fedden/RenderMan).

Après vérification, les méthodes RenderMan utilisées par SpinVAE2 (`RenderEngine`,
`load_plugin`, `set_patch`, `render_patch`, `get_audio_frames`,
`PatchGenerator.get_random_patch`) ont toutes un équivalent DawDreamer.

## Étapes de migration

### 1. Plugin Dexed recompilé en VST3

L'ancien `Dexed.so` (VST2, v0.9.4) posait deux problèmes sur les nœuds CPU : tentative de
connexion à un serveur X (display graphique inexistant en headless) et version obsolète. La
solution a été de **recompiler Dexed (v1.0.x) en VST3 directement sur le cluster** via cmake.
Cela règle d'un coup la compatibilité système (binaire lié aux bibliothèques locales) et le
mode headless (géré par les versions JUCE récentes). Le plugin est installé dans
`~/.vst3/Dexed.vst3`.

### 2. Décalage d'indices de paramètres

Point critique. Le VST3 Dexed expose un paramètre `MonoMode` à l'index 3, absent du schéma
SpinVAE2 (155 paramètres, indices 0–154). Le mapping retenu :

| Index SpinVAE2 | Index VST3                          |
|----------------|-------------------------------------|
| 0, 1, 2        | 0, 1, 2 (Cutoff, Resonance, Output) |
| i ≥ 3          | i + 1 (saute MonoMode)              |

Vérifié paramètre par paramètre. Sans ce remapping, tous les presets de
`dexed_presets.sqlite` auraient été décalés et le son faux.

### 3. Adaptation du code

Création d'un module wrapper `synth/dawdreamer_engine.py` qui imite l'API RenderMan (mêmes
méthodes, remapping d'index interne, conversion stéréo → mono). Dans `synth/dexed.py`,
seulement deux modifications : l'import et le chemin du plugin (`.vst3`). **Le reste de
SpinVAE2 est inchangé** — aucune autre partie du projet n'appelait directement RenderMan.

## Résultat

Pipeline fonctionnel de bout en bout : chargement du plugin, application d'un preset, rendu
d'une note MIDI, downsampling 48 kHz → 16 kHz, audio mono correct. Plus aucune dépendance à
RenderMan, Boost ou Python 2.

## Points à retenir

- **`LD_LIBRARY_PATH`** : le VST3 est lié au `libstdc++` de conda. Tout lancement de SpinVAE2
  doit inclure
  `export LD_LIBRARY_PATH=/data/anasynth_nonbp/manaconda3/lib:$LD_LIBRARY_PATH`.
  À intégrer au script de lancement.
- **Environnement** : venv `spinvae_2_18_env` (Python 3.10, basé sur conda `tf2.18`), avec
  `dawdreamer` installé.
- **Validation audio recommandée** : comparer le rendu de quelques presets connus entre ancien
  et nouveau moteur avant de régénérer un dataset complet.
- **Backup** : `synth/dexed.py.renderman_backup` conserve la version RenderMan d'origine.