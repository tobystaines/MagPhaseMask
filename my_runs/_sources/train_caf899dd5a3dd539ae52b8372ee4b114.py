import datetime
import os
import errno
import math
import tensorflow as tf


def train(sess, model, model_config, model_folder, handle, training_iterator, training_handle, validation_iterator,
          validation_handle, writer):
    """
    Train an audio_models.py model.
    """

    def validation(last_val_cost, min_val_cost, min_val_check, worse_val_checks, model, val_check):
        """
        Perform a validation check using the validation dataset.
        """
        print('Validating')
        sess.run(validation_iterator.initializer)
        val_costs = list()
        val_iteration = 1
        print('{ts}:\tEntering validation loop'.format(ts=datetime.datetime.now()))
        while True:
            try:
                val_cost = sess.run(model.cost, {model.is_training: False, handle: validation_handle})
                if val_iteration % 200 == 0:
                    print("{ts}:\tValidation iteration: {i}, Loss: {vc}".format(ts=datetime.datetime.now(),
                                                                                i=val_iteration, vc=val_cost))
                if math.isnan(val_cost):
                    print('Error: cost = nan\nDiscarding batch')
                else:
                    val_costs.append(val_cost)
                    val_iteration += 1
            except tf.errors.OutOfRangeError:
                # Calculate and record mean loss over validation dataset
                val_check_mean_cost = sum(val_costs) / len(val_costs)
                print('Validation check mean loss: {l}'.format(l=val_check_mean_cost))
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='Validation_mean_loss', simple_value=val_check_mean_cost)])
                writer.add_summary(summary, val_check)
                # If validation loss has worsened increment the counter, else, reset the counter
                if val_check_mean_cost > last_val_cost:
                    worse_val_checks += 1
                    print('Validation loss has worsened. worse_val_checks = {w}'.format(w=worse_val_checks))
                else:
                    worse_val_checks = 0
                    print('Validation loss has improved!')
                if val_check_mean_cost < min_val_cost:
                    min_val_cost = val_check_mean_cost
                    print('New best validation cost!')
                    min_val_check = val_check
                last_val_cost = val_check_mean_cost

                break

        return last_val_cost, min_val_cost, min_val_check, worse_val_checks

    def checkpoint(model_config, model_folder, saver, sess, global_step):
        """
        Take a checkpoint of the model.
        """
        # Make sure there is a folder to save the checkpoint in
        checkpoint_path = os.path.join(model_config["model_base_dir"], model_folder)
        try:
            os.makedirs(checkpoint_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        print('Checkpoint')
        saver.save(sess, os.path.join(checkpoint_path, model_folder), global_step=int(global_step))
        return os.path.join(checkpoint_path, model_folder + '-' + str(global_step))

    print('Starting training')
    # Initialise variables and define summary
    epoch = 0
    iteration = 1
    last_val_cost = 1
    min_val_cost = 1
    min_val_check = None
    val_check = 1
    worse_val_checks = 0
    latest_checkpoint_path = os.path.join(model_config['model_base_dir'], model_config['checkpoint_to_load'])
    # Summaries
    cost_summary = tf.summary.scalar('Training_loss', model.cost)
    mix_0_summary = tf.summary.image('Mixture_0', tf.expand_dims(model.mixed_input[:, :, :, 0], axis=3))
    voice_0_summary = tf.summary.image('Voice_0', tf.expand_dims(model.voice_input[:, :, :, 0], axis=3))
    mask_0_summary = tf.summary.image('Voice_Mask_0', tf.expand_dims(model.voice_mask[:, :, :, 0], axis=3))
    gen_voice_0_summary = tf.summary.image('Generated_Voice_0', tf.expand_dims(model.gen_voice[:, :, :, 0], axis=3))
    if model_config['data_type'] != 'mag':
        mix_1_summary = tf.summary.image('Mixture_1', tf.expand_dims(model.mixed_input[:, :, :, 1], axis=3))
        voice_1_summary = tf.summary.image('Voice_1', tf.expand_dims(model.voice_input[:, :, :, 1], axis=3))
        mask_1_summary = tf.summary.image('Voice_Mask_1', tf.expand_dims(model.voice_mask[:, :, :, 1], axis=3))
        gen_voice_1_summary = tf.summary.image('Generated_Voice_1', tf.expand_dims(model.gen_voice[:, :, :, 1], axis=3))
    if model_config['data_type'] in ['mag_real_imag', 'mag_phase_real_imag', 'mag_phase2', 'complex_to_mag_phase']:
        mix_2_summary = tf.summary.image('Mixture_2', tf.expand_dims(model.mixed_input[:, :, :, 2], axis=3))
        voice_2_summary = tf.summary.image('Voice_2', tf.expand_dims(model.voice_input[:, :, :, 2], axis=3))
    if model_config['data_type'] in ['mag_real_imag']:
        mask_2_summary = tf.summary.image('Voice_Mask_2', tf.expand_dims(model.voice_mask[:, :, :, 2], axis=3))
        gen_voice_2_summary = tf.summary.image('Generated_Voice_2', tf.expand_dims(model.gen_voice[:, :, :, 2], axis=3))
    if model_config['data_type'] in ['mag_phase_real_imag', 'complex_to_mag_phase']:
        mix_3_summary = tf.summary.image('Mixture_3', tf.expand_dims(model.mixed_input[:, :, :, 3], axis=3))
        voice_3_summary = tf.summary.image('Voice_3', tf.expand_dims(model.voice_input[:, :, :, 3], axis=3))
    if 'mag_' in model_config['data_type']:
        mag_loss_summary = tf.summary.scalar('Training_magnitude_loss', model.mag_loss)
    if 'phase' in model_config['data_type']:
        phase_loss_summary = tf.summary.scalar('Training_phase_loss', model.phase_loss)
    if 'real' in model_config['data_type'] and model_config['data_type'] != 'mag_phase_real_imag':
        real_loss_summary = tf.summary.scalar('Training_real_loss', model.real_loss)
    if 'imag' in model_config['data_type'] and model_config['data_type'] != 'mag_phase_real_imag':
        imag_loss_summary = tf.summary.scalar('Training_imag_loss', model.imag_loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, write_version=tf.train.SaverDef.V2)
    sess.run(training_iterator.initializer)
    # Begin training loop
    # Train for the specified number of epochs, unless early stopping is triggered
    while epoch < model_config['epochs'] and worse_val_checks < model_config['num_worse_val_checks']:
        try:
            try:
                if model_config['data_type'] == 'mag':
                    _, cost, cost_sum, mix_0, \
                        voice_0, mask_0, gen_voice_0 = sess.run([model.train_op, model.cost, cost_summary,
                                                                 mix_0_summary, voice_0_summary, mask_0_summary,
                                                                 gen_voice_0_summary], {model.is_training: True,
                                                                                        handle: training_handle})
                elif model_config['data_type'] in ['mag_phase', 'mag_phase_diff']:
                    _, cost, cost_sum, mag_loss_sum, phase_loss_sum, \
                        mix_0, voice_0, mask_0, gen_voice_0, mix_1, \
                        voice_1, mask_1, gen_voice_1 = sess.run([model.train_op, model.cost, cost_summary,
                                                                 mag_loss_summary, phase_loss_summary, mix_0_summary,
                                                                 voice_0_summary, mask_0_summary, gen_voice_0_summary,
                                                                 mix_1_summary, voice_1_summary, mask_1_summary,
                                                                 gen_voice_1_summary],
                                                                {model.is_training: True, handle: training_handle})
                elif model_config['data_type'] in ['mag_phase2']:
                    _, cost, cost_sum, mag_loss_sum, \
                        phase_loss_sum, mix_0, voice_0, mask_0, \
                        gen_voice_0, mix_1, voice_1, mask_1, \
                        gen_voice_1, mix_2,  voice_2 = sess.run([model.train_op, model.cost, cost_summary,
                                                                 mag_loss_summary, phase_loss_summary, mix_0_summary,
                                                                 voice_0_summary, mask_0_summary, gen_voice_0_summary,
                                                                 mix_1_summary, voice_1_summary, mask_1_summary,
                                                                 gen_voice_1_summary, mix_2_summary, voice_2_summary],
                                                                {model.is_training: True, handle: training_handle})
                elif model_config['data_type'] == 'real_imag':
                    _, cost, cost_sum, real_loss_sum, imag_loss_sum, mix_0, \
                        voice_0, mask_0, gen_voice_0, mix_1, \
                        voice_1, mask_1, gen_voice_1 = sess.run([model.train_op, model.cost, cost_summary,
                                                                 real_loss_summary, imag_loss_summary,
                                                                 mix_0_summary, voice_0_summary, mask_0_summary,
                                                                 gen_voice_0_summary, mix_1_summary, voice_1_summary,
                                                                 mask_1_summary, gen_voice_1_summary],
                                                                {model.is_training: True, handle: training_handle})
                elif model_config['data_type'] == 'mag_real_imag':
                    _, cost, cost_sum, mag_loss_sum, real_loss_sum, imag_loss_sum, mix_0, \
                        voice_0, mask_0, gen_voice_0, mix_1, \
                        voice_1, mask_1, gen_voice_1, mix_2, \
                        voice_2, mask_2, gen_voice_2  = sess.run([model.train_op, model.cost, cost_summary,
                                                                 mag_loss_summary, real_loss_summary, imag_loss_summary,
                                                                 mix_0_summary, voice_0_summary, mask_0_summary,
                                                                 gen_voice_0_summary, mix_1_summary, voice_1_summary,
                                                                 mask_1_summary, gen_voice_1_summary, mix_2_summary,
                                                                 voice_2_summary, mask_2_summary, gen_voice_2_summary],
                                                                {model.is_training: True, handle: training_handle})
                elif model_config['data_type'] in ['mag_phase_real_imag', 'complex_to_mag_phase']:
                    _, cost, cost_sum, mag_loss_sum, phase_loss_sum, \
                        mix_0, voice_0, mask_0, gen_voice_0, mix_1, \
                        voice_1, mask_1, gen_voice_1, \
                        mix_2, voice_2, mix_3, voice_3, = sess.run([model.train_op, model.cost, cost_summary,
                                                                    mag_loss_summary, phase_loss_summary,
                                                                    mix_0_summary, voice_0_summary, mask_0_summary,
                                                                    gen_voice_0_summary, mix_1_summary, voice_1_summary,
                                                                    mask_1_summary, gen_voice_1_summary, mix_2_summary,
                                                                    voice_2_summary, mix_3_summary, voice_3_summary],
                                                                   {model.is_training: True, handle: training_handle})

            except RuntimeWarning:
                print('Invalid value encountered. Ignoring batch.')
                continue

            if math.isnan(cost):
                print('Error: cost = nan')
                print('Loading latest checkpoint')
                restorer = tf.train.Saver()
                restorer.restore(sess, latest_checkpoint_path)
                break
            writer.add_summary(cost_sum, iteration)  # Record the loss at each iteration
            if 'mag_' in model_config['data_type']:
                writer.add_summary(mag_loss_sum, iteration)
            if 'phase' in model_config['data_type']:
                writer.add_summary(phase_loss_sum, iteration)
            if 'real' in model_config['data_type'] and model_config['data_type'] != 'mag_phase_real_imag':
                writer.add_summary(real_loss_sum, iteration)
            if 'imag' in model_config['data_type'] and model_config['data_type'] != 'mag_phase_real_imag':
                writer.add_summary(imag_loss_sum, iteration)
            if iteration % 200 == 0:
                print("{ts}:\tTraining iteration: {i}, Loss: {c}".format(ts=datetime.datetime.now(),
                                                                         i=iteration, c=cost))
            # If saving by iterations, take a checkpoint
            if model_config['saving'] and not model_config['save_by_epochs'] \
                    and iteration % model_config['save_iters'] == 0:
                latest_checkpoint_path = checkpoint(model_config, model_folder, saver, sess, iteration)

            # If using early stopping by iterations, enter validation loop
            if model_config['early_stopping'] and not model_config['val_by_epochs'] \
                    and iteration % model_config['val_iters'] == 0:
                last_val_cost, min_val_cost, min_val_check, worse_val_checks = validation(last_val_cost,
                                                                                          min_val_cost,
                                                                                          min_val_check,
                                                                                          worse_val_checks,
                                                                                          model,
                                                                                          val_check)
                val_check += 1

            iteration += 1

        # When the dataset is exhausted, note the end of the epoch
        except tf.errors.OutOfRangeError:
            print('{ts}:\tEpoch {e} finished after {i} iterations.'.format(ts=datetime.datetime.now(),
                                                                           e=epoch, i=iteration))
            try:
                writer.add_summary(mix_0, iteration)
                writer.add_summary(voice_0, iteration)
                writer.add_summary(mask_0, iteration)
                writer.add_summary(gen_voice_0, iteration)
                if model_config['data_type'] != 'mag':
                    writer.add_summary(mix_1, iteration)
                    writer.add_summary(voice_1, iteration)
                    writer.add_summary(mask_1, iteration)
                    writer.add_summary(gen_voice_1, iteration)
                if model_config['data_type'] in ['mag_real_imag', 'mag_phase_real_imag',
                                                 'mag_phase2', 'complex_to_mag_phase']:
                    writer.add_summary(mix_2, iteration)
                    writer.add_summary(voice_2, iteration)
                if model_config['data_type'] in ['mag_real_imag']:
                    writer.add_summary(mask_2, iteration)
                    writer.add_summary(gen_voice_2, iteration)
                if model_config['data_type'] in ['mag_phase_real_imag', 'complex_to_mag_phase']:
                    writer.add_summary(mix_3, iteration)
                    writer.add_summary(voice_3, iteration)
            except NameError:  # Indicates the try has not been successfully executed at all
                print('No images to record')
                break
            epoch += 1
            # If using early stopping by epochs, enter validation loop
            if model_config['early_stopping'] and model_config['val_by_epochs'] and iteration > 1:
                last_val_cost, min_val_cost, min_val_check, worse_val_checks = validation(last_val_cost,
                                                                                          min_val_cost,
                                                                                          min_val_check,
                                                                                          worse_val_checks,
                                                                                          model,
                                                                                          val_check)
                val_check += 1
            if model_config['saving'] and model_config['save_by_epochs']:
                latest_checkpoint_path = checkpoint(model_config, model_folder, saver, sess, epoch)
            sess.run(training_iterator.initializer)
    print('Training complete after {e} epochs.'.format(e=epoch))
    if model_config['early_stopping'] and worse_val_checks >= model_config['num_worse_val_checks']:
        print('Stopped early due to validation criteria.')
    else:
        # Final validation check
        if (iteration % model_config['val_iters'] != 1 or not model_config['early_stopping']) \
                and not model_config['val_by_epochs']:
            last_val_cost, min_val_cost, min_val_check, _ = validation(last_val_cost, min_val_cost, min_val_check,
                                                                       worse_val_checks, model, val_check)
    print('Finished requested number of epochs.')
    print('Final validation loss: {lvc}'.format(lvc=last_val_cost))
    if last_val_cost == min_val_cost:
        print('This was the best validation loss achieved')
    else:
        print('Best validation loss ({mvc}) achieved at validation check {mvck}'.format(mvc=min_val_cost,
                                                                                        mvck=min_val_check))
    return model

