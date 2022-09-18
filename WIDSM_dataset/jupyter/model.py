import keras
from conv_with_associated import Conv1d_with_associated

layers = keras.layers


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

    def create_model(self):
        """
        :param is_classify: whether to output only one label for a window
        :return: create model object
        """
        input_ = layers.Input(shape=[self.input_length, 3])
        
        x = layers.Lambda(lambda x: x)(input_)
    


        if self.use_init_batch_norm:
            x = layers.BatchNormalization(axis=-1)(x)

        # CONV EARLY FUSE
        x = Conv1d_with_associated(x, self.early_fuse_cnn_channels, self.early_fuse_conv_kernel_size,
                                       padding='same',
                                       droprate=self.early_conv_drop_rate, use_spatial_drop=self.use_spatial_drop)



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

        # MID FC
        x = layers.Dropout(self.drop_rate_before_midfc)(x)
        x = layers.TimeDistributed(layers.Dense(self.tcn_encoder_channels[-1], activation='relu'))(x)

        # DECODER
        for index, channel in enumerate(self.tcn_decoder_channels):
            x = layers.UpSampling1D(self.upsample_size[index])(x)
            x = Conv1d_with_associated(x, channel, self.tcn_kernel_size, padding='same', droprate=self.tcn_drop_rate,
                                       use_spatial_drop=self.use_spatial_drop)

        # LAST FC
        x = layers.Dropout(self.drop_rate_before_fcclassifier)(x)
        output_ = layers.TimeDistributed(layers.Dense(self.n_classes, activation='softmax'))(x)

        output_ = layers.GlobalAveragePooling1D()(output_)

        return keras.Model(inputs=input_, outputs=output_)

    def receptive_field(self):
        """
        :return: model tcn part's receptive field
        """
        x = 1
        for i in range(len(self.tcn_encoder_channels)):
            x *= self.tcn_encoder_maxpool_size[i]
        return self.tcn_kernel_size * (x - 1) + 1
