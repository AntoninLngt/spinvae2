"""Test du pipeline de rendu DawDreamer sur un petit lot de presets (mono-process)."""
import os
os.environ.setdefault('LD_LIBRARY_PATH', '/data/anasynth_nonbp/manaconda3/lib')

import config
from data import dataset

mc, tc = config.ModelConfig(), config.TrainConfig()
config.update_dynamic_config_params(mc, tc)

# Instancie le vrai DexedDataset, mais sans vérifier les contraintes (pas encore de fichiers rendus)
ds = dataset.DexedDataset(
    **dataset.model_config_to_dataset_kwargs(mc),
    restrict_to_labels=None,
    check_constrains_consistency=False,
)
print(ds)
print("Storage path:", ds.audio_storage_path)

# Limite à 20 presets pour le test
ds.valid_preset_UIDs = ds.valid_preset_UIDs[:20]

# Crée le dossier de sortie + fichier de contraintes
import shutil
if os.path.exists(ds.audio_storage_path):
    shutil.rmtree(ds.audio_storage_path)
ds.audio_storage_path.mkdir(parents=True, exist_ok=True)
ds.write_audio_render_constraints_file()

# Rend les wav pour ces 20 presets (mono-process via _generate_wav_files_batch)
from datetime import datetime
t0 = datetime.now()
ds._generate_wav_files_batch(ds.valid_preset_UIDs)
dt = (datetime.now() - t0).total_seconds()

# Vérifie ce qui a été écrit
import glob
wavs = glob.glob(str(ds.audio_storage_path.joinpath("*.wav")))
print(f"\n{len(wavs)} fichiers wav écrits en {dt:.1f}s ({1000*dt/max(len(wavs),1):.0f} ms/fichier)")
print("Exemples:", [os.path.basename(w) for w in wavs[:4]])
