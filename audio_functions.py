import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf


def normalise_audio(audio):
    """
    Nomralises an ndarray to the interval[-1 1]. For use on 1 dimensional audio waveforms (although will work on higher
    dimensional arrays as well).
    """
    norm_audio = 2*((audio - audio.min())/(audio.max()-audio.min())) - 1
    return norm_audio


def read_audio_py(py_path, sample_rate):
    mono, native_sr = sf.read(py_path)
    if native_sr != sample_rate:
        mono = librosa.core.resample(mono, native_sr, sample_rate)
    return np.expand_dims(mono, 1).astype(np.float32)


def read_audio(path, sample_rate, n_channels=1):

    return tf.py_func(read_audio_py, [path, sample_rate], tf.float32, stateful=False)


def read_audio_triple(path_a, path_b, path_c, sample_rate):
    """
    Takes in the path of three audio files and the required output sample rate,
    returns a tuple of tensors of the wave form of the audio files.
    """
    return (tf.py_func(read_audio_py, [path_a, sample_rate], tf.float32, stateful=False),
            tf.py_func(read_audio_py, [path_b, sample_rate], tf.float32, stateful=False),
            tf.py_func(read_audio_py, [path_c, sample_rate], tf.float32, stateful=False))


def compute_spectrogram(audio, n_fft, fft_hop, normalise):
    """
    Parameters
    ----------
    audio : mono audio shaped (n_samples, )
    n_fft: number of samples in each Fourier transform
    fft_hop: hop length between the start of Fourier transforms

    Returns
    -------
    Tensor of shape (n_frames, 1 + n_fft / 2, 4), where the last dimension is (real number, imaginary number, magnitude, phase)
    """

    def stft(x, normalise):
        spec = librosa.stft(
            x, n_fft=n_fft, hop_length=fft_hop, window='hann')
        mag = np.abs(spec)
        if normalise:
            mag = (mag - mag.min()) / (mag.max() - mag.min())

        return spec.real, spec.imag, mag, np.angle(spec)

    def mono_func(py_audio, normalise):
        real, imag, mag, phase = stft(py_audio[:, 0], normalise)
        ret = np.array([real, imag, mag, phase]).T
        return ret.astype(np.float32)

    with tf.name_scope('read_spectrogram'):
        ret = tf.py_func(mono_func, [audio, normalise], tf.float32, stateful=False)
        ret.set_shape([(audio.get_shape()[0].value/fft_hop) + 1, 1 + n_fft / 2, 4])
    return ret


def extract_spectrogram_patches(
        spec, n_fft, patch_window, patch_hop):
    """
    Parameters
    ----------
    spec : Spectrogram of shape (n_frames, 1 + n_fft / 2, 2)

    Returns
    -------
    Tensor of shape (n_patches, patch_window, 1 + n_fft / 2, 2)
        containing patches from spec.
    """
    with tf.name_scope('extract_spectrogram_patches'):
        spec4d = tf.expand_dims(spec, 0)

        patches = tf.extract_image_patches(
            spec4d, ksizes=[1, patch_window, 1 + n_fft / 2, 1],
            strides=[1, patch_hop, 1 + n_fft / 2, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        num_patches = tf.shape(patches)[1]

        return tf.reshape(patches, [num_patches, patch_window,
                                    int(1 + n_fft / 2), 2])


def extract_audio_patches(audio, fft_hop, patch_window, patch_hop):
    """
    Parameters
    ----------
    audio : Waveform audio of shape (n_samples, )

    Returns
    -------
    Tensor of shape (n_patches, patch_window) containing patches from audio.
    """
    with tf.name_scope('extract_audio_patches'):
        audio4d = tf.expand_dims(tf.expand_dims(audio, 0), 0)
        patch_length = (patch_window - 1) * fft_hop
        patch_hop_length = (patch_hop - 1) * fft_hop

        patches = tf.extract_image_patches(
            audio4d, ksizes=[1, 1, patch_length, 1],
            strides=[1, 1, patch_hop_length, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        num_patches = tf.shape(patches)[2]

        return tf.squeeze(tf.reshape(patches, [num_patches, 1, patch_length, 1]), 1)


def compute_spectrogram_map(audio_a, audio_b, audio_c, n_fft, fft_hop, normalise=False):
    """
    Takes three waveform arrays and return the corresponding spectrograms, and the original arrays.
    """
    spec_a = compute_spectrogram(audio_a, n_fft, fft_hop, normalise)
    spec_b = compute_spectrogram(audio_b, n_fft, fft_hop, normalise)
    spec_c = compute_spectrogram(audio_c, n_fft, fft_hop, normalise)

    return spec_a, spec_b, spec_c, audio_a, audio_b, audio_c


def extract_audio_patches_map(audio_a, audio_b, audio_c, fft_hop, patch_window, patch_hop):
    """
    Take three audio waveform arrays and split them each into overlapping patches of matching shape.
    """
    audio_patches_a = extract_audio_patches(audio_a, fft_hop, patch_window, patch_hop)
    audio_patches_b = extract_audio_patches(audio_b, fft_hop, patch_window, patch_hop)
    audio_patches_c = extract_audio_patches(audio_c, fft_hop, patch_window, patch_hop)

    return audio_patches_a, audio_patches_b, audio_patches_c


def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize, phaseIterations=0, phase=None, length=None):
    """
    From Stoller et al (2017)
    Computes an audio signal from the given magnitude spectrogram, and optionally an initial phase.
    Griffin-Lim is executed to recover/refine the given the phase from the magnitude spectrogram.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param phase: If given, starts ISTFT with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    """
    if phase is not None:
        if phaseIterations > 0:
            # Refine audio given initial phase with a number of iterations
            return reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations, phase, length)
        # reconstructing the new complex matrix
        stftMatrix = magnitude * np.exp(phase * 1j) # magnitude * e^(j*phase)
        audio = librosa.istft(stftMatrix, hop_length=hopSize, length=length)
    else:
        audio = reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations)
    return audio


def reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations=0, initPhase=None, length=None):
    """
    From Stoller et al (2017)
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram, optionally with a given
    intial phase.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param initPhase: If given, starts reconstruction with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    """
    for i in range(phaseIterations):
        if i == 0:
            if initPhase is None:
                reconstruction = np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                reconstruction = np.exp(initPhase * 1j) # e^(j*phase), so that angle => phase
        else:
            reconstruction = librosa.stft(audio, fftWindowSize, hopSize)[:reconstruction.shape[0],:reconstruction.shape[1]] # Indexing to keep the output the same size
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.istft(spectrum, hopSize, length=length)
        else:
            audio = librosa.istft(spectrum, hopSize)
    return audio


def zip_tensor_slices(*args):
    """
    Parameters
    ----------
    *args : list of _n_ _k_-dimensional tensors, where _k_ >= 2
        The first dimension has _m_ elements.

    Returns
    -------
    result : Dataset of _m_ examples, where each example has _n_
        records of _k_ - 1 dimensions.

    Example
    -------
    ds = (
        tf.data.Dataset.zip((
            tf.data.Dataset.from_tensors([[1,2], [3,4], [5, 6]]),
            tf.data.Dataset.from_tensors([[10, 20], [30, 40], [50, 60]])
        ))
        .flat_map(zip_tensor_slices)  # <--- *HERE*
    )
    el = ds.make_one_shot_iterator().get_next()
    print sess.run(el)
    print sess.run(el)

    # Output:
    # (array([1, 2], dtype=int32), array([10, 20], dtype=int32))
    # (array([3, 4], dtype=int32), array([30, 40], dtype=int32))
    """
    return tf.data.Dataset.zip(tuple([
        tf.data.Dataset.from_tensor_slices(arg)
        for arg in args
    ]))