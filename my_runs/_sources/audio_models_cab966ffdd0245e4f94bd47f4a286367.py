import tensorflow as tf
import model_functions as mf


class MagnitudeModel(object):
    """
    Top level object for models.
    Attributes:
        mixed_input: 4D tensor ([batch_size, height, width, channels]) - Input placeholder for mixed signals (voice plus background noise) - X
        voice_input: 4D tensor ([batch_size, height, width, channels]) - Input placeholder for isolated voice signal - Y
        mixed_phase: 4D tensor ([batch_size, height, width, 1]) - Input placeholder for phase spectrogram of mixed signals (voice plus background noise)
        mixed_audio: 3D tensor ([batch_size, length, 1]) - Input placeholder for waveform audio of mixed signals (voice plus background noise)
        voice_audio: 3D tensor ([batch_size, length, 1]) - Input placeholder for waveform audio of isolated voice signal
        background_audio: 3D tensor ([batch_size, length, 1]) - Input placeholder for waveform audio of background noise signal
        is_training: Boolean - should the model be trained on the current input or not
        learning_rate: The learning rate the model should be trained with.
        data_type: The type of data representations to be used in the model (' mag', 'mag_phase', 'mag_phase2', 'mag_phase_diff', 'real_imag', 'mag_real_imag')
        phase_weight: The weight applied to phase loss relative to magnitude loss
        name: Model instance name
    """
    def __init__(self, mixed_input, voice_input, mixed_phase, mixed_audio, voice_audio, background_audio,
                 is_training, learning_rate, data_type, phase_weight, phase_loss_masking,
                 phase_loss_approximation, name):
        with tf.variable_scope(name):
            self.mixed_input = mixed_input
            self.voice_input = voice_input
            self.mixed_phase = mixed_phase
            self.mixed_audio = mixed_audio
            self.voice_audio = voice_audio
            self.background_audio = background_audio
            self.is_training = is_training

            # Initialise the selected model variant
            if data_type == 'complex_to_mag_phase':
                self.voice_mask_network = UNet(mixed_input[:, :, :, 0:2], data_type, is_training=is_training, reuse=False, name='voice-mask-unet')
            else:
                self.voice_mask_network = UNet(mixed_input, data_type, is_training=is_training, reuse=False, name='voice-mask-unet')

            self.voice_mask = self.voice_mask_network.output

            # Depending on the data_type, setup the loss functions and optimisation
            if data_type == 'mag':
                self.gen_voice = self.voice_mask * mixed_input
                self.cost = mf.l1_loss(self.gen_voice, voice_input)

            elif data_type == 'mag_phase':
                self.gen_voice = self.voice_mask * mixed_input
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1],
                                                   phase_loss_masking, phase_loss_approximation,
                                                   self.gen_voice[:, :, :, 0]) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss)/2

            elif data_type == 'mag_phase_diff2':
                self.gen_voice_mag = tf.expand_dims(self.voice_mask[:, :, :, 0] * mixed_input[:, :, :, 0], axis=3)
                self.mag_loss = mf.l1_loss(self.gen_voice_mag[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(mf.phase_difference(mixed_input[:, :, :, 1], voice_input[:, :, :, 1]),
                                                   self.voice_mask[:, :, :, 1], phase_loss_masking,
                                                   phase_loss_approximation, self.gen_voice_mag) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss) / 2
                self.gen_voice_phase = tf.expand_dims(self.voice_mask[:, :, :, 1] + mixed_input[:, :, :, 1], axis=3)
                self.gen_voice = mf.concat(self.gen_voice_mag, self.gen_voice_phase)

            elif data_type == 'mag_phase_diff':
                self.gen_voice_mag = tf.expand_dims(self.voice_mask[:, :, :, 0] * mixed_input[:, :, :, 0], axis=3)
                self.gen_voice_phase = tf.expand_dims(self.voice_mask[:, :, :, 1] + mixed_input[:, :, :, 1], axis=3)
                self.gen_voice = mf.concat(self.gen_voice_mag, self.gen_voice_phase)
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice_phase, voice_input[:, :, :, 1],
                                                   phase_loss_masking, phase_loss_approximation,
                                                   self.gen_voice_mag) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss) / 2

            elif data_type == 'real_imag':
                self.gen_voice = self.voice_mask * mixed_input
                self.real_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.imag_loss = mf.l1_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1])
                self.cost = (self.real_loss + self.imag_loss)/2

            elif data_type == 'mag_real_imag':
                self.gen_voice = self.voice_mask * mixed_input
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.real_loss = mf.l1_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1])
                self.imag_loss = mf.l1_loss(self.gen_voice[:, :, :, 2], voice_input[:, :, :, 2])
                self.cost = (self. mag_loss + self.real_loss + self.imag_loss) / 3

            elif data_type == 'mag_phase2':
                self.mag_mask = self.voice_mask[:, :, :, 0]
                self.phase_mask = tf.angle(tf.complex(self.voice_mask[:, :, :, 1], self.voice_mask[:, :, :, 2]))
                self.voice_mask = mf.concat(tf.expand_dims(self.mag_mask, axis=3), tf.expand_dims(self.phase_mask, axis=3))
                self.gen_voice_mag = self.mag_mask * mixed_input[:, :, :, 0]
                self.gen_voice_phase = self.phase_mask * tf.squeeze(mixed_phase, axis=3)
                self.voice_phase = tf.angle(tf.complex(self.voice_input[:, :, :, 1], self.voice_input[:, :, :, 2]))
                self.gen_voice = mf.concat(tf.expand_dims(self.gen_voice_mag, axis=3),
                                           tf.expand_dims(self.gen_voice_phase, axis=3))
                self.mag_loss = mf.l1_loss(self.gen_voice_mag, voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice_phase, self.voice_phase, phase_loss_masking,
                                                   phase_loss_approximation, self.gen_voice_mag) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss) / 2

            elif data_type == 'mag_phase_real_imag':
                self.gen_voice = self.voice_mask * mixed_input[:, :, :, 2:4]
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 2])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 3],
                                                   phase_loss_masking, phase_loss_approximation,
                                                   self.gen_voice[:, :, :, 0]) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss)/2

            elif data_type == 'complex_to_mag_phase':
                self.gen_voice = self.voice_mask * mixed_input[:, :, :, 2:4]
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 2])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 3],
                                                   phase_loss_masking, phase_loss_approximation,
                                                   self.gen_voice[:, :, :, 0]) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss) / 2

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.5,
            )
            self.train_op = self.optimizer.minimize(self.cost)


class UNet(object):
    """
    Magnitude model U-Net
    """
    def __init__(self, input_tensor, data_type, is_training, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            self.encoder = UNetEncoder(input_tensor, is_training, reuse)
            self.decoder = UNetDecoder(self.encoder.output, self.encoder, data_type, is_training, reuse)
            self.output = mf.tanh(self.decoder.output) / 2 + .5


class UNetEncoder(object):
    """
    The down-convolution side of a convoltional U-Net model.
    """

    def __init__(self, input_tensor, is_training, reuse):

        self.input_tensor = input_tensor
        with tf.variable_scope('encoder'):
            with tf.variable_scope('layer-1'):
                net = mf.conv(self.input_tensor, filters=16, kernel_size=5, stride=(2, 2))
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=32, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=64, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l3 = net

            with tf.variable_scope('layer-4'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=128, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l4 = net

            with tf.variable_scope('layer-5'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=256, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l5 = net

            with tf.variable_scope('layer-6'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=512, kernel_size=5, stride=(2, 2))

            self.output = net


class UNetDecoder(object):
    """
    The up-convolution side of a convolutional U-Net model
    """
    def __init__(self, input_tensor, encoder, data_type, is_training, reuse):
        self.input_tensor = input_tensor

        with tf.variable_scope('decoder'):
            with tf.variable_scope('layer-1'):
                net = mf.relu(self.input_tensor)
                net = mf.deconv(net, filters=256, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                net = mf.dropout(net, .5)
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = mf.relu(mf.concat(net, encoder.l5))
                net = mf.deconv(net, filters=128, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                net = mf.dropout(net, .5)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = mf.relu(mf.concat(net, encoder.l4))
                net = mf.deconv(net, filters=64, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                net = mf.dropout(net, .5)
                self.l3 = net

            with tf.variable_scope('layer-4'):
                net = mf.relu(mf.concat(net, encoder.l3))
                net = mf.deconv(net, filters=32, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l4 = net

            with tf.variable_scope('layer-5'):
                net = mf.relu(mf.concat(net, encoder.l2))
                net = mf.deconv(net, filters=16, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l5 = net

            with tf.variable_scope('layer-6'):
                if data_type == 'mag_phase_real_imag':
                    self.out_depth = 2
                else:
                    self.out_depth = encoder.input_tensor.shape[3]
                net = mf.relu(mf.concat(net, encoder.l1))
                net = mf.deconv(net, filters=self.out_depth, kernel_size=5, stride=(2, 2))

            self.output = net
