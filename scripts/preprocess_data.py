import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(phishing_path, legitimate_path, max_length, vocab_size):
    # Load phishing and legitimate data
    phishing_data = pd.read_csv(phishing_path)
    legitimate_data = pd.read_csv(legitimate_path)

    # Assign labels (1 for phishing, 0 for legitimate)
    phishing_data['label'] = 1
    legitimate_data['label'] = 0
    data = pd.concat([phishing_data, legitimate_data])
    data = shuffle(data).reset_index(drop=True)

    # Extract URLs and labels
    urls = data['url'].values
    labels = data['label'].values

    # Tokenize URLs
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(urls)
    url_sequences = tokenizer.texts_to_sequences(urls)
    url_sequences = tf.keras.preprocessing.sequence.pad_sequences(url_sequences, maxlen=max_length, padding='post')

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(url_sequences, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, tokenizer
