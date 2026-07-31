import io
import os
import sys
from typing import Any, Dict, Tuple

from pydantic import BaseModel, Field
import soundfile as sf

from adtool.utils.expose_config.expose_config import expose

# spinvae2 is a separate repo, not an installed package -- add it to sys.path so we can
# reuse its Dexed/DawDreamer rendering engine instead of duplicating it here.
SPINVAE2_ROOT = os.path.expanduser("~/projects/spinvae2")
if SPINVAE2_ROOT not in sys.path:
    sys.path.insert(0, SPINVAE2_ROOT)

from synth.dexed import Dexed  # noqa: E402


class GenerationParams(BaseModel):
    output_Fs: int = Field(16000, ge=8000, le=48000)
    midi_note_duration_s: float = Field(3.0, ge=0.1, le=10.0)
    render_duration_s: float = Field(4.0, ge=0.1, le=15.0)
    # Reference note used for the whole spinvae2 dataset -- keep identical here so that
    # discovered presets remain directly comparable to the human-preset baseline.
    midi_pitch: int = Field(56, ge=0, le=127)
    midi_velocity: int = Field(75, ge=1, le=127)


@expose
class DexedSimulation:
    """ Renders Dexed (DX7) presets to audio, via the DawDreamer-based engine already used to
    regenerate spinvae2's dataset (synth.dexed.Dexed, itself backed by
    synth.dawdreamer_engine). One MIDI note only (pitch/velocity fixed), matching the reference
    note used everywhere else in spinvae2. """

    config = GenerationParams

    def __init__(
        self,
        output_Fs: int,
        midi_note_duration_s: float,
        render_duration_s: float,
        midi_pitch: int,
        midi_velocity: int,
    ) -> None:
        self.output_Fs = output_Fs
        self.midi_pitch = midi_pitch
        self.midi_velocity = midi_velocity
        # Loading the Dexed VST3 is costly (~seconds) -- one engine instance is reused across
        # every map() call rather than recreated per-evaluation.
        self.dexed_engine = Dexed(
            output_Fs=output_Fs,
            midi_note_duration_s=midi_note_duration_s,
            render_duration_s=render_duration_s,
        )
        self.signal = None

    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        # DexedParameterMap.sample()/mutate() produce a preset in Dexed's native
        # "list of (idx, value) tuples" format, under dynamic_params["preset"].
        preset = input["params"]["dynamic_params"]["preset"]
        self.dexed_engine.assign_preset(preset)
        audio, _sr = self.dexed_engine.render_note(self.midi_pitch, self.midi_velocity, normalize=False)
        self.signal = audio

        input["output"] = self.signal
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:
        byte_signal = io.BytesIO()
        sf.write(byte_signal, self.signal, self.output_Fs, format="wav")
        return [(byte_signal.getvalue(), "wav")]
