from matplotlib.pyplot import cla
import tensorflow as tf
import tensorflow.keras.models as models
from .model_zoo.UNet import unet
from .model_zoo.DeepLabV3plus import DeeplabV3_plus
from .model_zoo.modify_DeepLabV3plus import deepLabV3Plus
from .model_zoo.EfficientNetV2 import EfficientNetV2M, EfficientNetV2S
from .model_zoo.DDRNet_23_slim import ddrnet_23_slim
from .model_zoo.mobileNetV3 import MobileNetV3_Small


def age_estimation_model(image_size, num_classes=2, loss_type='softmax'):
    if loss_type == 'sparse':
        classifier_activation = None
    else:
        classifier_activation = "softmax"

    model = EfficientNetV2S(input_shape=(image_size[0], image_size[1], 3),
                            num_classes=num_classes,
                            classifier_activation=classifier_activation)

    return model
