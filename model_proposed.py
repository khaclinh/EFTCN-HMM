import tensorflow.keras as keras
import tensorflow as tf
from conv_with_associated import Conv1d_with_associated

layers = keras.layers



from tensorflow.keras.layers import Layer
class Trainabble_Multiply(Layer): # a scaled layer
    def __init__(self, **kwargs):
        super(Trainabble_Multiply
    , self).__init__(**kwargs)
    def build(self, input_shape):
        self.output_dim = input_shape[1] 
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer='one', trainable=True)
        super(Trainabble_Multiply
    , self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        return tf.multiply(x, self.W)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        base_config = super(Trainabble_Multiply, self).get_config()
        return dict(list(base_config.items()))


class ED_TCN_fmap_fuse():
    def __init__(self,
                 n_classes,  # type: int
                 input_length=512,  # type: int
                 early_fuse_cnn_channels=64,  # type: int
                 before_encode_pool_size=4,
                 tcn_encoder_channels=(64,) * 4,  # type: list
                 tcn_decoder_channels=(64,) * 2,  # type: list
                 tcn_encoder_maxpool_size=(2,) * 7,
                 tcn_decoder_upsample_size=(5, 2),  # type: list
                 early_fuse_conv_kernel_size=5,
                 tcn_kernel_size=5,
                 early_conv_drop_rate=0.,
                 tcn_drop_rate=0.3,
                 use_spatial_drop=True,
                 drop_rate_before_midfc=0.5,
                 drop_rate_before_fcclassifier=0.7,
                 use_init_batch_norm=True,
                 use_last_fc=True,
                 ):
        self.n_classes = n_classes
        self.input_length = input_length

        self.early_fuse_cnn_channels = early_fuse_cnn_channels

        self.tcn_encoder_channels = tcn_encoder_channels

        self.tcn_decoder_channels = tcn_decoder_channels
        self.upsample_size = tcn_decoder_upsample_size

        self.use_init_batch_norm = use_init_batch_norm
        self.early_fuse_conv_kernel_size = early_fuse_conv_kernel_size
        self.before_encode_pool_size = before_encode_pool_size

        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_encoder_maxpool_size = tcn_encoder_maxpool_size

        self.early_conv_drop_rate = early_conv_drop_rate
        self.tcn_drop_rate = tcn_drop_rate
        self.use_spatial_drop = use_spatial_drop

        self.drop_rate_before_midfc = drop_rate_before_midfc
        self.use_last_fc = use_last_fc
        self.drop_rate_before_fcclassifier = drop_rate_before_fcclassifier

        # check number of element in input lists
        if len(self.tcn_encoder_maxpool_size) != len(self.tcn_encoder_channels):
            raise ValueError('max pool sizes and encoder layers not match!')
        if len(self.upsample_size) != len(self.tcn_decoder_channels):
            raise ValueError('upsample sizes and decoder layers not match!')
        if (self.tcn_encoder_channels[-1] % 8 != 0):
            raise ValueError('the number of channels before mid self-attention block must be divisible by 8')

    def create_model(self):
        """
        :param is_classify: whether to output only one label for a window
        :return: create model object
        """
        input_ = layers.Input(shape=[self.input_length, 18])
        
        hand_acc = layers.Lambda(lambda x: x[:, :, :3])(input_)
        hand_gyr = layers.Lambda(lambda x: x[:, :, 3:6])(input_)
        ankle_acc = layers.Lambda(lambda x: x[:, :, 6:9])(input_)
        ankle_gyr = layers.Lambda(lambda x: x[:, :, 9:12])(input_)
        chest_acc = layers.Lambda(lambda x: x[:, :, 12:15])(input_)
        chest_gyr = layers.Lambda(lambda x: x[:, :, 15:])(input_)


        if self.use_init_batch_norm:
            hand_acc = layers.BatchNormalization(axis=-1)(hand_acc)
            hand_gyr = layers.BatchNormalization(axis=-1)(hand_gyr)
            ankle_acc = layers.BatchNormalization(axis=-1)(ankle_acc)
            ankle_gyr = layers.BatchNormalization(axis=-1)(ankle_gyr)
            chest_acc = layers.BatchNormalization(axis=-1)(chest_acc)
            chest_gyr = layers.BatchNormalization(axis=-1)(chest_gyr)

        # CONV EARLY FUSE
        hand_acc = Conv1d_with_associated(hand_acc, self.early_fuse_cnn_channels, self.early_fuse_conv_kernel_size,
                                       padding='same',
                                       droprate=self.early_conv_drop_rate, use_spatial_drop=self.use_spatial_drop)
        hand_gyr = Conv1d_with_associated(hand_gyr, self.early_fuse_cnn_channels, self.early_fuse_conv_kernel_size,
                                       padding='same',
                                       droprate=self.early_conv_drop_rate, use_spatial_drop=self.use_spatial_drop)
        ankle_acc = Conv1d_with_associated(ankle_acc, self.early_fuse_cnn_channels, self.early_fuse_conv_kernel_size,
                                       padding='same',
                                       droprate=self.early_conv_drop_rate, use_spatial_drop=self.use_spatial_drop)
        ankle_gyr = Conv1d_with_associated(ankle_gyr, self.early_fuse_cnn_channels, self.early_fuse_conv_kernel_size,
                                       padding='same',
                                       droprate=self.early_conv_drop_rate, use_spatial_drop=self.use_spatial_drop)
        chest_acc = Conv1d_with_associated(chest_acc, self.early_fuse_cnn_channels, self.early_fuse_conv_kernel_size,
                                       padding='same',
                                       droprate=self.early_conv_drop_rate, use_spatial_drop=self.use_spatial_drop)
        chest_gyr = Conv1d_with_associated(chest_gyr, self.early_fuse_cnn_channels, self.early_fuse_conv_kernel_size,
                                       padding='same',
                                       droprate=self.early_conv_drop_rate, use_spatial_drop=self.use_spatial_drop)

        x = layers.Concatenate(axis=-1)([hand_acc, hand_gyr, ankle_acc, ankle_gyr, chest_acc, chest_gyr])

        x = layers.MaxPool1D(self.before_encode_pool_size)(x)

        # ENCODER
        for index, channel in enumerate(self.tcn_encoder_channels):
            this_drop_rate = self.tcn_drop_rate

            x = Conv1d_with_associated(x, channel, self.tcn_kernel_size, padding='same', droprate=this_drop_rate,
                                       use_spatial_drop=self.use_spatial_drop)
            mod = int(x.shape[1] % self.tcn_encoder_maxpool_size[index])
            if mod != 0:
                x = layers.ZeroPadding1D([0, self.tcn_encoder_maxpool_size[index] - mod])(x)
            x = layers.MaxPool1D(self.tcn_encoder_maxpool_size[index])(x)

        # MID Self-Attention Block
        
        n_timesteps = x.shape[1]
        n_channels = x.shape[2]
        
        x_reshaped = layers.Reshape((1, n_timesteps, n_channels))(x)
        
        key = layers.Conv2D(n_channels // 8, (1, 1))(x_reshaped)
        query = layers.Conv2D(n_channels // 8, (1, 1))(x_reshaped)
        value = layers.Conv2D(n_channels, (1, 1))(x_reshaped)
        
        att = layers.Attention()([query, value, key])
        
        att = Trainabble_Multiply()(att)
        x = layers.Add()([x, att])
        
        x = layers.Reshape((n_timesteps, n_channels))(x)

        # DECODER
        for index, channel in enumerate(self.tcn_decoder_channels):
            x = layers.UpSampling1D(self.upsample_size[index])(x)
            x = Conv1d_with_associated(x, channel, self.tcn_kernel_size, padding='same', droprate=self.tcn_drop_rate,
                                       use_spatial_drop=self.use_spatial_drop)

        # LAST FC
        x = layers.Dropout(self.drop_rate_before_fcclassifier)(x)
        output_ = layers.TimeDistributed(layers.Dense(12, activation='softmax'))(x)

        output_ = layers.GlobalAveragePooling1D()(output_)

        return tf.keras.Model(inputs=input_, outputs=output_)

    def receptive_field(self):
        """
        :return: model tcn part's receptive field
        """
        x = 1
        for i in range(len(self.tcn_encoder_channels)):
            x *= self.tcn_encoder_maxpool_size[i]
        return self.tcn_kernel_size * (x - 1) + 1
