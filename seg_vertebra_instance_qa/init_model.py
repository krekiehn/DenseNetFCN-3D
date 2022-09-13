# init a FCN Model for training and later inference?
from DenseNet3D import DenseNet3D_FCN

model_fcn = DenseNet3D_FCN((64, 64, 64, 1), nb_dense_block=5, growth_rate=16,
                               nb_layers_per_block=4, upsampling_type='upsampling', classes=1, activation='sigmoid')
model_fcn.summary()