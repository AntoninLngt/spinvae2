"""
DawDreamer-based replacement for the RenderMan (librenderman) engine.
Mimics the subset of the RenderMan API used by SpinVAE2's Dexed class:
  - RenderEngineWrapper: load_plugin, set_patch, render_patch, get_audio_frames,
    get_plugin_parameter_size, get_plugin_parameters_description
  - PatchGeneratorWrapper: get_random_patch

Index remapping (SpinVAE2 <-> Dexed VST3):
  SpinVAE2 presets use 155 params (indices 0..154), WITHOUT the 'MonoMode' param.
  The Dexed VST3 inserts 'MonoMode' at index 3, so:
      spinvae_idx 0,1,2  -> vst3_idx 0,1,2
      spinvae_idx i>=3   -> vst3_idx i+1
"""
import numpy as np
import dawdreamer as daw


N_SPINVAE_PARAMS = 155  # Dexed params as seen by SpinVAE2 (0..154)


def spinvae_to_vst3_idx(i: int) -> int:
    """ Maps a SpinVAE2 param index (0..154) to the Dexed VST3 param index. """
    return i if i < 3 else i + 1


class RenderEngineWrapper:
    """ Mimics rm.RenderEngine using DawDreamer under the hood. """

    def __init__(self, sample_rate, buffer_size, fft_size=512):
        self.sample_rate = int(sample_rate)
        self.buffer_size = int(buffer_size)
        self.fft_size = fft_size  # unused, kept for API compatibility
        self._engine = daw.RenderEngine(self.sample_rate, self.buffer_size)
        self._synth = None
        self._last_audio = None  # numpy array (mono), filled by render_patch

    def load_plugin(self, plugin_path, _unused_index=0):
        """ Loads the Dexed VST3. Second arg kept for RenderMan API compatibility. """
        self._synth = self._engine.make_plugin_processor("dexed", str(plugin_path))
        self._engine.load_graph([(self._synth, [])])

    def set_patch(self, preset):
        """ :param preset: list of (spinvae_idx, value) tuples, values in [0, 1]. """
        for idx, value in preset:
            self._synth.set_parameter(spinvae_to_vst3_idx(int(idx)), float(value))

    def get_patch(self):
        """ Returns the current patch as a list of (spinvae_idx, value) tuples. """
        return [(i, self._synth.get_parameter(spinvae_to_vst3_idx(i)))
                for i in range(N_SPINVAE_PARAMS)]

    def render_patch(self, midi_note, midi_velocity, note_duration_s, render_duration_s):
        """ Renders one MIDI note with the currently-set patch. """
        self._synth.clear_midi()
        self._synth.add_midi_note(int(midi_note), int(midi_velocity), 0.0, float(note_duration_s))
        self._engine.render(float(render_duration_s))
        audio = self._engine.get_audio()  # shape (channels, samples)
        # RenderMan returned mono; Dexed outputs identical L/R -> take channel 0
        self._last_audio = np.asarray(audio[0], dtype=np.float32)

    def get_audio_frames(self):
        """ Returns the last rendered audio as a 1D float array (mono). """
        if self._last_audio is None:
            return np.zeros(0, dtype=np.float32)
        return self._last_audio

    def get_plugin_parameter_size(self):
        return N_SPINVAE_PARAMS

    def get_plugin_parameters_description(self):
        """ Returns [(spinvae_idx, name), ...] for the 155 Dexed params. """
        descr = []
        for i in range(N_SPINVAE_PARAMS):
            vst3_idx = spinvae_to_vst3_idx(i)
            descr.append((i, self._synth.get_parameter_name(vst3_idx)))
        return descr


class PatchGeneratorWrapper:
    """ Mimics rm.PatchGenerator. SpinVAE2 only uses get_random_patch() to count params. """

    def __init__(self, engine: RenderEngineWrapper):
        self._engine = engine

    def get_random_patch(self):
        """ Returns a random patch as list of (idx, value) tuples (values in [0,1]). """
        rng = np.random.default_rng()
        return [(i, float(rng.random())) for i in range(N_SPINVAE_PARAMS)]

# Aliases for RenderMan API compatibility
RenderEngine = RenderEngineWrapper
PatchGenerator = PatchGeneratorWrapper

