from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model
