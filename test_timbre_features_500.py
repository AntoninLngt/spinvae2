"""Test TT + ACTM sur 500 presets, pour verifier la selection de features vs papier SpinVAE2."""
import config
from data import dataset

if __name__ == "__main__":
    mc, tc = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(mc, tc)

    ds = dataset.DexedDataset(
        **dataset.model_config_to_dataset_kwargs(mc),
        restrict_to_labels=None,
        check_constrains_consistency=False,
    )

    ds.valid_preset_UIDs = ds.valid_preset_UIDs[:500]
    print(f"Test sur {len(ds.valid_preset_UIDs)} presets")

    ds.compute_and_store_timbre_features(default_midi_note_only=True)

    print("\n=== Verification ===")
