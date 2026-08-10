import io
import os
import pickle
import sys
from typing import Any, Dict, Tuple

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf

from adtool.systems.System import System
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
class DexedSimulation(System):
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
        super().__init__()
        self.output_Fs = output_Fs
        self.midi_note_duration_s = midi_note_duration_s
        self.render_duration_s = render_duration_s
        self.midi_pitch = midi_pitch
        self.midi_velocity = midi_velocity
        self.signal = None
        self._dexed_engine = None

    @property
    def dexed_engine(self) -> Dexed:
        # Lazily built and cached rather than eagerly loaded in __init__: adtool restores
        # checkpoints via `cls.__new__(cls)` + `__dict__.update(state)`, bypassing __init__
        # entirely, and DawDreamer's RenderEngine wraps a native C++ object that cannot be
        # pickled -- so it can never be part of the checkpointed state (see checkpoint_state()
        # below) and must always be rebuilt on first use instead, whichever path created us.
        if self._dexed_engine is None:
            self._dexed_engine = Dexed(
                output_Fs=self.output_Fs,
                midi_note_duration_s=self.midi_note_duration_s,
                render_duration_s=self.render_duration_s,
            )
        return self._dexed_engine

    def checkpoint_state(self) -> Dict[str, Any]:
        # adtool's checkpoint store calls this hook (see
        # adtool.utils.persistence.checkpoint.Checkpoint._checkpoint_state) instead of going
        # through pickle's own __getstate__/__setstate__ protocol. `_dexed_engine` (DawDreamer's
        # native RenderEngine) can never be pickled -- drop anything that isn't, defensively,
        # rather than only the one attribute known to be unpicklable today.
        state = {}
        for key, value in self.__dict__.items():
            try:
                pickle.dumps(value)
            except Exception:
                value = None
            state[key] = value
        return state

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

        # adtool's shipped viewer only accepts .mp4/.png as a discovery's visual thumbnail
        # (VALID_VISUAL_SUFFIXES in adtool.user_tools.visu.coordinates) -- a .wav-only
        # discovery has no resolvable "visual" and gets silently dropped from the map
        # entirely. A waveform plot gives the viewer something to show; the .wav is still
        # saved alongside it (both are named "visu.<ext>") for actual listening.
        # Uses matplotlib's object-oriented API (Figure + FigureCanvasAgg) directly rather than
        # pyplot -- pyplot would need `matplotlib.use("Agg")` for a headless experiment run,
        # but that call mutates the backend for the whole process, which breaks inline plotting
        # if this module ever gets imported into an interactive Jupyter kernel (as the overview
        # notebook does). The Figure/Canvas API needs no backend selection at all.
        fig = Figure(figsize=(4, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(np.asarray(self.signal), linewidth=0.5, color="steelblue")
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        byte_image = io.BytesIO()
        FigureCanvasAgg(fig).print_png(byte_image)

        return [(byte_image.getvalue(), "png"), (byte_signal.getvalue(), "wav")]
