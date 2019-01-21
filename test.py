import datetime
import os
import pickle
import math
import tensorflow as tf


def test(sess, model, model_config, handle, testing_iterator, testing_handle, test_count, experiment_id):
    """
    Test an audio_models.py model, saving the outputs to a pickle file.
    """
    # Create a folder for saving pickle files
    dump_folder = 'dumps/' + str(experiment_id)
    if not os.path.isdir(dump_folder):
        os.mkdir(dump_folder)
    print('Starting testing')
    sess.run(testing_iterator.initializer)
    iteration = 0
    test_costs = list()

    print('{ts}:\tEntering test loop'.format(ts=datetime.datetime.now()))
    # For each batch, save the data required for metric calculation and record the loss.
    while True:
        try:
            cost, voice_est_matrix, voice_ref_audio, background_audio,\
                mixed_audio, mixed_input, mixed_phase = sess.run([model.cost, model.gen_voice, model.voice_audio,
                                                                  model.background_audio, model.mixed_audio,
                                                                  model.mixed_input, model.mixed_phase],
                                                                 {model.is_training: False, handle: testing_handle})
            results = (cost, voice_est_matrix, voice_ref_audio, background_audio,
                       mixed_audio, mixed_input, mixed_phase, model_config)
            if math.isnan(cost):
                print('Error: cost = nan\nDiscarding batch')
            else:
                test_costs.append(cost)
                dump_name = dump_folder + '/test_count_' + str(test_count) + '_iteration_' + str(iteration)
                pickle.dump(results, open(dump_name, 'wb'))
                if iteration % 200 == 0:
                    print("{ts}:\tTesting iteration: {i}, Loss: {c}".format(ts=datetime.datetime.now(),
                                                                            i=iteration, c=cost))
                iteration += 1
        except tf.errors.OutOfRangeError:
            # At the end of the dataset, calculate, record and print mean loss
            mean_cost = sum(test_costs) / len(test_costs)
            print('Test pass complete\n'
                  'Mean loss over test set: {}\n'
                  'Data saved to {} for later audio metric calculation'.format(mean_cost, dump_folder))
            break
    test_count += 1

    return mean_cost, test_count
