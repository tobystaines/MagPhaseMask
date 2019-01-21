import tensorflow as tf
import model_functions as mf
from SegCaps import capsule_layers
from keras import layers


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
        variant: The type of model ('unet', capsunet', basic_capsnet', 'basic_convnet')
        is_training: Boolean - should the model be trained on the current input or not
        learning_rate: The learning rate the model should be trained with.
        data_type: The type of data representations to be used in the model (' mag', 'mag_phase', 'mag_phase2', 'mag_phase_diff', 'real_imag', 'mag_real_imag')
        phase_weight: The weight applied to phase loss relative to magnitude loss
        name: Model instance name
    """
    def __init__(self, mixed_input, voice_input, mixed_phase, mixed_audio, voice_audio, background_audio,
                 variant, is_training, learning_rate, data_type, phase_weight, name):
        with tf.variable_scope(name):
            self.mixed_input = mixed_input
            self.voice_input = voice_input
            self.mixed_phase = mixed_phase
            self.mixed_audio = mixed_audio
            self.voice_audio = voice_audio
            self.background_audio = background_audio
            self.variant = variant
            self.is_training = is_training

            # Initialise the selected model variant
            if self.variant in ['unet', 'capsunet'] and data_type == 'complex_to_mag_phase':
                self.voice_mask_network = UNet(mixed_input[:, :, :, 0:2], variant, data_type, is_training=is_training, reuse=False, name='voice-mask-unet')
            elif self.variant in ['unet', 'capsunet']:
                self.voice_mask_network = UNet(mixed_input, variant, data_type, is_training=is_training, reuse=False, name='voice-mask-unet')
            elif self.variant == 'basic_capsnet':
                self.voice_mask_network = BasicCapsNet(mixed_input, name='basic_capsnet')
            elif self.variant == 'basic_convnet':
                self.voice_mask_network = BasicConvNet(mixed_input, is_training=is_training, reuse=None, name='basic_convnet')

            self.voice_mask = self.voice_mask_network.output

            # Depending on the data_type, setup the loss functions and optimisation
            if data_type == 'mag':
                self.gen_voice = self.voice_mask * mixed_input
                self.cost = mf.l1_loss(self.gen_voice, voice_input)

            elif data_type == 'mag_phase':
                self.gen_voice = self.voice_mask * mixed_input
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1]) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss)/2

            elif data_type == 'mag_phase_diff2':
                self.gen_voice_mag = tf.expand_dims(self.voice_mask[:, :, :, 0] * mixed_input[:, :, :, 0], axis=3)
                self.mag_loss = mf.l1_loss(self.gen_voice_mag[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(mf.phase_difference(mixed_input[:, :, :, 1], voice_input[:, :, :, 1]),
                                                   self.voice_mask[:, :, :, 1]) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss) / 2
                self.gen_voice_phase = tf.expand_dims(self.voice_mask[:, :, :, 1] + mixed_input[:, :, :, 1], axis=3)
                self.gen_voice = mf.concat(self.gen_voice_mag, self.gen_voice_phase)

            elif data_type == 'mag_phase_diff':
                self.gen_voice_mag = tf.expand_dims(self.voice_mask[:, :, :, 0] * mixed_input[:, :, :, 0], axis=3)
                self.gen_voice_phase = tf.expand_dims(self.voice_mask[:, :, :, 1] + mixed_input[:, :, :, 1], axis=3)
                self.gen_voice = mf.concat(self.gen_voice_mag, self.gen_voice_phase)
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1]) * phase_weight
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
                self.gen_mag = self.mag_mask * mixed_input[:, :, :, 0]
                self.gen_phase = self.phase_mask * tf.squeeze(mixed_phase, axis=3)
                self.voice_phase = tf.angle(tf.complex(self.voice_input[:, :, :, 1], self.voice_input[:, :, :, 2]))
                self.gen_voice = mf.concat(tf.expand_dims(self.gen_mag, axis=3), tf.expand_dims(self.gen_phase, axis=3))
                self.mag_loss = mf.l1_loss(self.gen_mag, voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(self.gen_phase, self.voice_phase) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss) / 2

            elif data_type == 'mag_phase_real_imag':
                self.gen_voice = self.voice_mask * mixed_input[:, :, :, 2:4]
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 2])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 3]) * phase_weight
                self.cost = (self.mag_loss + self.phase_loss)/2

            elif data_type == 'complex_to_mag_phase':
                self.gen_voice = self.voice_mask * mixed_input[:, :, :, 2:4]
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 2])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 3]) * phase_weight
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
    def __init__(self, input_tensor, variant, data_type, is_training, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            self.variant = variant

            if self.variant == 'unet':
                self.encoder = UNetEncoder(input_tensor, is_training, reuse)
                self.decoder = UNetDecoder(self.encoder.output, self.encoder, data_type, is_training, reuse)
            elif self.variant == 'capsunet':
                self.encoder = CapsUNetEncoder(input_tensor, is_training, reuse)
                self.decoder = CapsUNetDecoder(self.encoder.output, self.encoder, is_training, reuse)

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


class CapsUNetEncoder(object):
    """
    The down-convolutional side of a capsule based U-Net model.
    """

    def __init__(self, input_tensor, is_training, reuse):
        # net = layers.Input(shape=input_tensor)
        self.input_tensor = input_tensor
        with tf.variable_scope('Encoder'):
            with tf.variable_scope('Convolution'):
                # Layer 1: A conventional Conv2D layer
                net = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu',
                                    name='conv1')(self.input_tensor)
                self.conv1 = net

                # Reshape layer to be 1 capsule x [filters] atoms
                _, H, W, C = net.get_shape()
                net = layers.Reshape((H.value, W.value, 1, C.value))(net)

            # Layer 1: Primary Capsule: Conv cap with routing 1
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=8, strides=2, padding='same',
                                                  routings=1, name='primarycaps')(net)
            self.primary_caps = net

            # Layer 2: Convolutional Capsules
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=8, strides=2, padding='same',
                                                  routings=3, name='conv_cap_2')(net)
            self.conv_cap_2 = net

            # Layer 3: Convolutional Capsules
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=16, strides=2, padding='same',
                                                  routings=3, name='conv_cap_3')(net)
            self.conv_cap_3 = net

            # Layer 4: Convolutional Capsules
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=2, padding='same',
                                                  routings=3, name='conv_cap_4')(net)

            self.output = net


class CapsUNetDecoder(object):
    """
    The up-convolutional side of a capsule based U-Net model.
    """

    def __init__(self, input_tensor, encoder, is_training, reuse):
        net = input_tensor
        with tf.variable_scope('Decoder'):
            # Layer 1 Up: Deconvolutional capsules, skip connection, convolutional capsules
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=16, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_1')(net)
            self.upcap_1 = net

            net = layers.Concatenate(axis=-2, name='skip_1')([net, encoder.conv_cap_3])

            # Layer 2 Up: Deconvolutional capsules, skip connection, convolutional capsules
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=8, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_2')(net)
            self.upcap_2 = net

            net = layers.Concatenate(axis=-2, name='skip_2')([net, encoder.conv_cap_2])

            # Layer 3 Up: Deconvolutional capsules, skip connection
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=8, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_3')(net)
            self.upcap_3 = net

            net = layers.Concatenate(axis=-2, name='skip_3')([net, encoder.primary_caps])

            # Layer 4 Up: Deconvolutional capsules, skip connection
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=16, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_4')(net)
            self.upcap_4 = net

            # Reconstruction - Reshape, skip connection + 3x conventional Conv2D layers
            _, H, W, C, D = net.get_shape()

            net = layers.Reshape((H.value, W.value, D.value))(net)
            net = layers.Concatenate(axis=-1, name='skip_4')([net, encoder.conv1])

            net = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(net)

            net = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(net)

            if tf.rank(encoder.input_tensor) == 3:
                self.out_depth = 1
            else:
                self.out_depth = encoder.input_tensor.shape[3].value

            net = layers.Conv2D(filters=self.out_depth, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='sigmoid', name='out_recon')(net)

            self.output = net


class BasicCapsNet(object):

    def __init__(self, input_tensor, name):
        """
        A basic capsule network operating on magnitude spectrograms.
        """
        with tf.variable_scope(name):
            self.input_tensor = input_tensor
            if tf.rank(self.input_tensor) == 3:
                self.out_depth = 1
            else:
                self.out_depth = input_tensor.shape[3].value

            with tf.variable_scope('layer_1'):
                net = mf.conv(input_tensor, filters=128, kernel_size=5, stride=(1, 1))

                # Reshape layer to be 1 capsule x [filters] atoms
                _, H, W, C = net.get_shape()
                net = layers.Reshape((H.value, W.value, 1, C.value))(net)
                self.conv1 = net

            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=16, strides=1,
                                                  padding='same',
                                                  routings=1, name='layer_2')(net)
            self.primary_caps = net

            net = capsule_layers.ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1,
                                                  padding='same',
                                                  routings=3, name='layer_3')(net)
            self.seg_caps = net

            net = capsule_layers.ConvCapsuleLayer(kernel_size=1, num_capsule=self.out_depth, num_atoms=1, strides=1,
                                                  padding='same',
                                                  routings=3, name='mask')(net)
            net = tf.squeeze(net, -1)

            self.output = net


class BasicConvNet(object):
    def __init__(self, input_tensor, is_training, reuse, name):
        """
        input_tensor: Tensor with shape [batch_size, height, width, channels]
        is_training:  Boolean - should the model be trained on the current input or not
        name:         Model instance name
        """
        with tf.variable_scope(name):
            self.input_tensor = input_tensor
            if tf.rank(self.input_tensor) == 3:
                self.out_depth = 1
            else:
                self.out_depth = input_tensor.shape[3].value

            with tf.variable_scope('layer_1'):
                net = mf.relu(input_tensor)
                net = mf.conv(net, filters=128, kernel_size=5, stride=(1, 1))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l1 = net

            with tf.variable_scope('layer_2'):
                net = mf.relu(net)
                net = mf.conv(net, filters=128, kernel_size=5, stride=(1, 1))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l2 = net

            with tf.variable_scope('layer_3'):
                net = mf.relu(net)
                net = mf.conv(net, filters=16, kernel_size=5, stride=(1, 1))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l3 = net

            with tf.variable_scope('mask'):
                net = mf.relu(net)
                net = mf.conv(net, filters=self.out_depth, kernel_size=5, stride=(1, 1))
                self.voice_mask = net

            self.output = net
