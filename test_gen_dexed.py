"""Test du pipeline gen_dexed_dataset sur un petit lot (wav uniquement, mono-process)."""
import glob, os
import config
from data import dataset
from data.regenerate import _gen_dataset

mc, tc = config.ModelConfig(), config.TrainConfig()
config.update_dynamic_config_params(mc, tc)

ds = dataset.DexedDataset(
    **dataset.model_config_to_dataset_kwargs(mc),
    restrict_to_labels=None,
    check_constrains_consistency=False,
)
ds.valid_preset_UIDs = ds.valid_preset_UIDs[:20]
print(ds)
print("Storage:", ds.audio_storage_path)

_gen_dataset(ds, regenerate_wav=True, regenerate_spectrograms=False, try_read_dataset=False)

wavs = glob.glob(str(ds.audio_storage_path.joinpath("*.wav")))
print(f"\n{len(wavs)} wav ecrits. Exemples: {[os.path.basename(w) for w in wavs[:4]]}")
