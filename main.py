import tensorflow as tf
from tensorflow.python import debug as tf_debug
from sacred import Experiment
from sacred.observers import FileStorageObserver

import os
import datetime
import json

import audio_models
import dataset
from train import train
from test import test


ex = Experiment('UNet_Speech_Separation', interactive=True)
ex.observers.append(FileStorageObserver.create('my_runs'))


@ex.config
def cfg():
    model_config = {'data_type': 'mag_phase',  # From [' mag', 'mag_phase', 'mag_phase_diff', 'real_imag',
                                                    # 'mag_real_imag', 'complex_to_mag_phase']
                    'phase_weight': 0.0005,  # When using a model which learns to estimate phase, defines how much
                                            # weight phase loss should be given against magnitude loss
                    'initialisation_test': False,  # Whether or not to calculate test metrics before training
                    'completion_test': False,  # Whether ot not to calculate test metrics after training
                    'loading': False,  # Whether to load an existing checkpoint
                    'training': False,
                    'checkpoint_to_load': "37/37-7",  # Checkpoint format: run/run-step
                    'saving': True,  # Whether to take checkpoints
                    'save_by_epochs': True,  # Checkpoints at end of each epoch or every 'save_iters'?
                    'save_iters': 10000,  # Number of training iterations between checkpoints
                    'early_stopping': True,  # Should validation data checks be used for early stopping?
                    'val_by_epochs': True,  # Validation at end of each epoch or every 'val_iters'?
                    'val_iters': 50000,  # Number of training iterations between validation checks,
                    'num_worse_val_checks': 3,  # Number of successively worse validation checks before early stopping,
                    'sample_rate': 16000,  # Desired sample rate of audio. Input will be resampled to this
                    'n_fft': 1024,  # Number of samples in each fourier transform
                    'fft_hop': 256,  # Number of samples between the start of each fourier transform
                    'n_parallel_readers': 16,
                    'patch_window': 256,  # Number of fourier transforms (rows) in each patch
                    'patch_hop': 128,  # Number of fourier transforms between the start of each patch
                    'batch_size': 50,  # Number of patches in each batch
                    'n_shuffle': 1000,  # Number of patches buffered before batching
                    'learning_rate': 0.0001,  # The learning rate to be used by the model
                    'epochs': 16,  # Number of full passes through the dataset to train for
                    'normalise_mag': True,  # Are magnitude spectrograms normalised in pre-processing?
                    'GPU': '0',
                    'phase_loss_masking': False,
                    'phase_loss_approximation': False,
                    'loss_function': 'l1_phase_loss',
                    'chime_data_root': '/vol/data/NewCHiME/',
                    'model_base_dir': '/home/enterprise.internal.city.ac.uk/acvn728/MagPhaseMask/checkpoints',
                    'log_dir':'logs/ssh'
                    }


@ex.automain
def do_experiment(model_config):

    if not model_config['training']:
        config_file_loc = f'my_runs/{model_config["checkpoint_to_load"].split("/")[0]}/config.json'
        with open(config_file_loc) as config_file:
            model_config = json.load(config_file)
            model_config['training'] = False
            model_config['loading'] = True
            model_config['completion_test'] = True

    tf.reset_default_graph()
    experiment_id = ex.current_run._id
    print('Experiment ID: {eid}'.format(eid=experiment_id))

    # Prepare data
    print('Preparing dataset')
    train_data, val_data, test_data = dataset.prepare_datasets(model_config)
    print('Dataset ready')

    # Start session
    tf_config = tf.ConfigProto()
    #tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.visible_device_list = str(model_config['GPU'])
    sess = tf.Session(config=tf_config)
    #sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")

    print('Session started')

    # Create data iterators
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
    mixed_spec, voice_spec, background_spec, mixed_audio, voice_audio, background_audio = iterator.get_next()

    training_iterator = train_data.make_initializable_iterator()
    validation_iterator = val_data.make_initializable_iterator()
    testing_iterator = test_data.make_initializable_iterator()

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    testing_handle = sess.run(testing_iterator.string_handle())
    print('Iterators created')

    # Create variable placeholders and model
    is_training = tf.placeholder(shape=(), dtype=bool)
    mixed_phase = tf.expand_dims(mixed_spec[:, :, :-1, 3], 3)

    print('Creating model')
    # Restructure data from pipeline based on data type required
    if model_config['data_type'] == 'mag':
        mixed_input = tf.expand_dims(mixed_spec[:, :, :-1, 2], 3)
        voice_input = tf.expand_dims(voice_spec[:, :, :-1, 2], 3)
    elif model_config['data_type'] in ['mag_phase', 'mag_phase_diff']:
        mixed_input = mixed_spec[:, :, :-1, 2:4]
        voice_input = voice_spec[:, :, :-1, 2:4]
    elif model_config['data_type'] == 'real_imag':
        mixed_input = mixed_spec[:, :, :-1, 0:2]
        voice_input = voice_spec[:, :, :-1, 0:2]
    elif model_config['data_type'] in ['mag_real_imag', 'mag_phase2']:
        mixed_input = tf.concat([tf.expand_dims(mixed_spec[:, :, :-1, 2], 3), mixed_spec[:, :, :-1, 0:2]], 3)
        voice_input = tf.concat([tf.expand_dims(voice_spec[:, :, :-1, 2], 3), voice_spec[:, :, :-1, 0:2]], 3)
    elif model_config['data_type'] in ['mag_phase_real_imag', 'complex_to_mag_phase']:
        mixed_input = mixed_spec[:, :, :-1, :]
        voice_input = voice_spec[:, :, :-1, :]

    model = audio_models.MagnitudeModel(mixed_input, voice_input, mixed_phase, mixed_audio, voice_audio, background_audio,
                                        is_training, model_config['learning_rate'], model_config['data_type'],
                                        model_config['phase_weight'], model_config['phase_loss_masking'],
                                        model_config['phase_loss_approximation'], 'Magnitude_Model',
                                        model_config['loss_function'])

    sess.run(tf.global_variables_initializer())

    if model_config['loading']:
        print('Loading checkpoint')
        checkpoint = os.path.join(model_config['model_base_dir'], model_config['checkpoint_to_load'])
        restorer = tf.train.Saver()
        restorer.restore(sess, checkpoint)

    # Summaries
    model_folder = str(experiment_id)
    writer = tf.summary.FileWriter(os.path.join(model_config["log_dir"], model_folder), graph=sess.graph)

    # Get baseline metrics at initialisation
    test_count = 0
    if model_config['initialisation_test']:
        print('Running initialisation test')
        initial_test_loss, test_count = test(sess, model, model_config, handle, testing_iterator, testing_handle,
                                             test_count, experiment_id)

    if model_config['training']:
        # Train the model
        model = train(sess, model, model_config, model_folder, handle, training_iterator, training_handle,
                      validation_iterator, validation_handle, writer)

    if model_config['completion_test']:
        # Test the trained model
        mean_test_loss, test_count = test(sess, model, model_config, handle, testing_iterator, testing_handle,
                                          test_count, experiment_id)
        if model_config['initialisation_test']:
            print('\tInitial test loss: {init}'.format(init=initial_test_loss))
        print('\tFinal test loss: {final}'.format(final=mean_test_loss))
    print('{ts}:\n\tAll done with experiment {exid}!'.format(ts=datetime.datetime.now(), exid=experiment_id))



