"""Test multiprocessing sur un petit lot."""
import os
os.environ['LD_LIBRARY_PATH'] = '/data/anasynth_nonbp/manaconda3/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import config
from data import dataset

mc, tc = config.ModelConfig(), config.TrainConfig()
config.update_dynamic_config_params(mc, tc)
ds = dataset.DexedDataset(**dataset.model_config_to_dataset_kwargs(mc),
                          restrict_to_labels=None, check_constrains_consistency=False)
# 60 presets pour forcer plusieurs workers
ds.valid_preset_UIDs = ds.valid_preset_UIDs[:60]
ds.generate_wav_files()  # passe par le chemin multiprocess (24 workers)

import glob
wavs = glob.glob(str(ds.audio_storage_path.joinpath("*.wav")))
print(f"\n{len(wavs)} wav écrits via multiprocessing")
