from functools import partial
import numpy as np
import tensorflow as tf
import audio_functions as af
import re
import os
from glob import glob


def zip_files(directory_a, directory_b, directory_c):
    """
    Takes in three directories (a, b and c) and returns an array, where each row is a triple of matching file paths,
    one from each directory, with directory a in col 0, directory b in col 1 an directory c in col 2.
    """

    filelist_a = [f for f in os.listdir(directory_a) if
                  os.path.isfile(os.path.join(directory_a, f)) and re.search('CH0', f) is None]
    filelist_b = [f for f in os.listdir(directory_b) if
                  os.path.isfile(os.path.join(directory_b, f)) and re.search('CH0', f) is None]

    zipped_list = list()

    for file_a in filelist_a:
        for file_b in filelist_b:
            if 'CHiME' in directory_a:
                if file_a[:13] == file_b[:13] and (file_a[17:] == file_b[17:] or len(file_a) != len(file_b)):
                    zipped_list.append((str(directory_a + '/' + file_a),
                                        str(directory_b + '/' + file_b),
                                        str(directory_c + '/' + file_a)))
                    if len(file_a) == len(file_b):
                        filelist_b.remove(file_b)
                        break
            else:
                if file_a == file_b:
                    zipped_list.append((str(directory_a + file_a),
                                        str(directory_b + file_b),
                                        str(directory_c + file_a)))
                    filelist_b.remove(file_b)
                    break

    if len(zipped_list) == 0:
        zipped_list = np.empty((0, 3))
    else:
        zipped_list = np.array(zipped_list)

    return zipped_list


def get_paired_dataset(zipped_files,
                       sample_rate,
                       n_fft,
                       fft_hop,
                       patch_window,
                       patch_hop,
                       n_parallel_readers,
                       batch_size,
                       n_shuffle,
                       normalise):
    """
    Returns a data pipeline (now tripple, rather than pair) of spectrogram and audio files
    """
    return (
        tf.data.Dataset.from_tensor_slices((zipped_files[:, 0], zipped_files[:, 1], zipped_files[:, 2]))
        .map(partial(af.read_audio_triple,
                     sample_rate=sample_rate),
             num_parallel_calls=n_parallel_readers)
        .map(partial(af.extract_audio_patches_map,
                     fft_hop=fft_hop,
                     patch_window=patch_window,
                     patch_hop=patch_hop,),
             num_parallel_calls=n_parallel_readers)
        .flat_map(af.zip_tensor_slices)
        .map(partial(af.compute_spectrogram_map,
                     n_fft=n_fft,
                     fft_hop=fft_hop,
                     normalise=normalise),
             num_parallel_calls=n_parallel_readers)
        .shuffle(n_shuffle).batch(batch_size).prefetch(3)
    )


def prepare_datasets(model_config):

    def build_datasets(model_config, root, path):
        train_files = zip_files(os.path.join(root, path['x_train']),
                                os.path.join(root, path['y_train_v']),
                                os.path.join(root, path['y_train_b']))
        train = get_paired_dataset(train_files,
                                   model_config['sample_rate'],
                                   model_config['n_fft'],
                                   model_config['fft_hop'],
                                   model_config['patch_window'],
                                   model_config['patch_hop'],
                                   model_config['n_parallel_readers'],
                                   model_config['batch_size'],
                                   model_config['n_shuffle'],
                                   model_config['normalise_mag'])

        val_files = zip_files(os.path.join(root, path['x_val']),
                              os.path.join(root, path['y_val_v']),
                              os.path.join(root, path['y_val_b']))
        val = get_paired_dataset(val_files,
                                 model_config['sample_rate'],
                                 model_config['n_fft'],
                                 model_config['fft_hop'],
                                 model_config['patch_window'],
                                 model_config['patch_hop'],
                                 model_config['n_parallel_readers'],
                                 model_config['batch_size'],
                                 model_config['n_shuffle'],
                                 model_config['normalise_mag'])

        test_files = zip_files(os.path.join(root, path['x_test']),
                               os.path.join(root, path['y_test_v']),
                               os.path.join(root, path['y_test_b']))
        test = get_paired_dataset(test_files,
                                  model_config['sample_rate'],
                                  model_config['n_fft'],
                                  model_config['fft_hop'],
                                  model_config['patch_window'],
                                  model_config['patch_hop'],
                                  model_config['n_parallel_readers'],
                                  model_config['batch_size'],
                                  model_config['n_shuffle'],
                                  model_config['normalise_mag'])
        return train, val, test

    if model_config['local_run']:  # If running on local machine, mini dataset is all in one folder
        path = {'x_train': 'train_sup/Mixed',
                'y_train': 'train_sup/Voice',
                'x_val': 'validation/Mixed',
                'y_val': 'validation/Voice',
                'x_test': 'test/Mixed',
                'y_test': 'test/Voice'}
        train_data, val_data, test_data = build_datasets(model_config, model_config['data_root'], path)
        return train_data, val_data, test_data

    else:  # If running on server, data is in several folders and requires concatenation
        if 'CHiME' in model_config['dataset']:
            # Get CHiME data
            sets = list()
            for string in ['bus', 'caf', 'ped', 'str']:
                path = {'x_train': 'tr05_' + string + '_simu/',
                        'y_train_v': 'tr05_org',
                        'y_train_b': 'tr05_' + string + '_bg/',
                        'x_val': 'dt05_' + string + '_simu/',
                        'y_val_v': 'dt05_bth',
                        'y_val_b': 'dt05_' + string + '_bg/',
                        'x_test': 'et05_' + string + '_simu/',
                        'y_test_v': 'et05_bth',
                        'y_test_b': 'et05_' + string + '_bg/'}
                sets.append(build_datasets(model_config, model_config['chime_data_root'], path))
            chime_train_data = sets[0][0].concatenate(sets[1][0].concatenate(sets[2][0].concatenate(sets[3][0])))
            chime_val_data = sets[0][1].concatenate(sets[1][1].concatenate(sets[2][1].concatenate(sets[3][1])))
            chime_test_data = sets[0][2].concatenate(sets[1][2].concatenate(sets[2][2].concatenate(sets[3][2])))

        if 'LibriSpeech' in model_config['dataset']:
            #### LibriSpeech data and code has not been updated to function was a triple (voice, background, mixed)
            #### dataset, so this section of code will cause errors.

            # Get list of LibriSpeech sub-directories
            voice_train_dirs = glob(model_config['librispeech_data_root'] + 'Voice/train-clean-100/**/', recursive=True)
            voice_val_dirs = glob(model_config['librispeech_data_root'] + 'Voice/dev-clean/**/', recursive=True)
            voice_test_dirs = glob(model_config['librispeech_data_root'] + 'Voice/test-clean/**/', recursive=True)

            mix_train_dirs = glob(model_config['librispeech_data_root'] + 'Mixed/train-clean-100/**/', recursive=True)
            mix_val_dirs = glob(model_config['librispeech_data_root'] + 'Mixed/dev-clean/**/', recursive=True)
            mix_test_dirs = glob(model_config['librispeech_data_root'] + 'Mixed/test-clean/**/', recursive=True)

            if ('LibriSpeech_m' or 'LibriSpeech_l') in model_config['dataset']:
                voice_train_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Voice/train-clean-360/**/', recursive=True))
                mix_train_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Mixed/train-clean-360/**/', recursive=True))

            if 'LibriSpeech_l' in model_config['dataset']:
                voice_train_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Voice/train-other-500/**/', recursive=True))
                voice_val_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Voice/dev-other/**/', recursive=True))
                voice_test_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Voice/test-other/**/', recursive=True))
                mix_train_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Mixed/train-other-500/**/', recursive=True))
                mix_val_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Mixed/dev-other/**/', recursive=True))
                mix_test_dirs.extend(
                    glob(model_config['librispeech_data_root'] + 'Mixed/test-other/**/', recursive=True))

            # Check corresponding list are of equal length
            assert len(voice_train_dirs) == len(mix_train_dirs)
            assert len(voice_val_dirs) == len(mix_val_dirs)
            assert len(voice_test_dirs) == len(mix_test_dirs)

            train_file_list = np.empty((0, 2))
            for i in range(len(voice_train_dirs)):
                train_file_list = np.concatenate((train_file_list, zip_files(mix_train_dirs[i], voice_train_dirs[i])), axis=0)
            libri_train_data = get_paired_dataset(train_file_list,
                                                  model_config['sample_rate'],
                                                  model_config['n_fft'],
                                                  model_config['fft_hop'],
                                                  model_config['patch_window'],
                                                  model_config['patch_hop'],
                                                  model_config['n_parallel_readers'],
                                                  model_config['batch_size'],
                                                  model_config['n_shuffle'],
                                                  model_config['normalise_mag'])

            val_file_list = np.empty((0, 2))
            for i in range(len(voice_val_dirs)):
                val_file_list = np.concatenate((val_file_list, zip_files(mix_val_dirs[i], voice_val_dirs[i])), axis=0)
            libri_val_data = get_paired_dataset(val_file_list,
                                                model_config['sample_rate'],
                                                model_config['n_fft'],
                                                model_config['fft_hop'],
                                                model_config['patch_window'],
                                                model_config['patch_hop'],
                                                model_config['n_parallel_readers'],
                                                model_config['batch_size'],
                                                model_config['n_shuffle'],
                                                model_config['normalise_mag'])

            test_file_list = np.empty((0, 2))
            for i in range(len(voice_test_dirs)):
                test_file_list = np.concatenate((test_file_list, zip_files(mix_test_dirs[i], voice_test_dirs[i])), axis=0)
            libri_test_data = get_paired_dataset(test_file_list,
                                                 model_config['sample_rate'],
                                                 model_config['n_fft'],
                                                 model_config['fft_hop'],
                                                 model_config['patch_window'],
                                                 model_config['patch_hop'],
                                                 model_config['n_parallel_readers'],
                                                 model_config['batch_size'],
                                                 model_config['n_shuffle'],
                                                 model_config['normalise_mag'])

        if model_config['dataset'] == 'CHiME':
            return chime_train_data, chime_val_data, chime_test_data
        elif model_config['dataset'] in ['LibriSpeech_s', 'LibriSpeech_m', 'LibriSpeech_l']:
            return libri_train_data, libri_val_data, libri_test_data
        else:
            return chime_train_data.concatenate(libri_train_data), \
                   chime_val_data.concatenate(libri_val_data), \
                   chime_test_data.concatenate(libri_test_data)
