import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

@register_keras_serializable()
class VGG_TF(tf.keras.Model):
    def __init__(self,num_classes,fine_tune_at=None,**kwargs):
        super(VGG_TF, self).__init__(**kwargs)
        # Get the pretrained VGG19 network (excluding the top classification layers)
        self.num_classes = num_classes
        self.vgg = VGG19(weights='imagenet', include_top=False)

        # Make the convolutional layers non-trainable by default (optional, for feature extraction)
        self.vgg.trainable = False

        # # Define the layers you want to access for CAM (similar to your PyTorch structure)
        self.features_conv = self.vgg.get_layer('block5_conv4') # Example: Access the output of the last conv layer

        # Define a max pooling layer (similar to your PyTorch structure)
        self.max_pool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Define a flatten layer
        self.flatten = layers.Flatten()

        # Define a classifier
        self.classifier = tf.keras.Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            # The number of output units will depend on your number of classes
            # You'll need to add this when you know your specific task.
            layers.Dense(num_classes, activation='softmax')
        ])

        # Placeholder for gradients (TensorFlow doesn't use hooks like PyTorch)
        self.gradients = None
        self.activations = None
        self.fine_tune_at = fine_tune_at
        self._set_trainable_layers()

    def _set_trainable_layers(self):
        if self.fine_tune_at is not None and self.fine_tune_at > 0:
            # Unfreeze the last 'fine_tune_at' convolutional layers
            for layer in self.vgg.layers[-self.fine_tune_at:]:
                if not isinstance(layer,
                                  layers.BatchNormalization):  # Important: Keep BatchNormalization frozen initially
                    layer.trainable = True

    def call(self, inputs):
        x = self.vgg(inputs)
        self.activations = self.features_conv.output # Store activations

        # Apply the remaining pooling
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self, tape, target_output):
        # This is a TensorFlow way to get gradients with respect to the activations
        return tape.gradient(target_output, self.activations)

    def get_activations(self, inputs):
        # We've stored the activations in the forward pass
        return self.activations

    def preprocess(self, image, label):
        # Resize the image to the input size expected by VGG19
        resized_image = tf.image.resize(image, (224, 224))
        # Apply the VGG19 specific preprocessing
        processed_image = preprocess_input(resized_image)
        return processed_image, label

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,  # Ensure this is 'num_classes'
            'fine_tune_at': self.fine_tune_at
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(num_classes=config['num_classes'],
                   fine_tune_at=config.get('fine_tune_at'),
                   name=config.get('name'),  # Handle the 'name' argument
                   dtype=config.get('dtype'),  # Handle the 'dtype' argument
                   trainable=config.get('trainable'),  # Handle the 'trainable' argument
                   **{})  # Pass any *other* config without duplicating known args# Pass any other config items as kwargs