import keras_hub
import tensorflow_datasets as tfds

classifier = keras_hub.models.TextClassifier.from_preset(
    "bert_base_en_uncased",
    activation="softmax",
    num_classes=2,
)

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train[:2000]", "test[:400]"],
    as_supervised=True,
    batch_size=32,
)

# Add training parameters to control time and prevent overfitting
classifier.fit(
    imdb_train.take(100),
    validation_data=imdb_test.take(20),
    epochs=3,  # BERT usually needs only 2-4 epochs
    verbose=1
)

preds = classifier.predict(["What an amazing movie!", "A total waste of time."])
print(preds)
