from keras.optimizers import Adam

from models.generator import build_generator
from models.discriminator import build_discriminator
from models.gan import build_gan
from scripts.preprocess_data import load_and_preprocess_data
import numpy as np


def run_training():
    # Hyperparameters
    vocab_size = 10000
    max_length = 100
    epochs = 10000
    batch_size = 64
    half_batch = batch_size // 2

    # Load data
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_data('data/phishing.csv',
                                                                           'data/legitimate.csv', max_length,
                                                                           vocab_size)

    # Build models
    generator = build_generator(vocab_size, max_length)
    discriminator = build_discriminator(vocab_size, max_length)

    # Compile the discriminator
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    gan = build_gan(generator, discriminator)

    # Training loop
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_urls = X_train[idx]

        noise = np.random.randint(0, vocab_size, (half_batch, max_length))
        gen_urls = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_urls, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_urls, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.randint(0, vocab_size, (batch_size, max_length))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")
