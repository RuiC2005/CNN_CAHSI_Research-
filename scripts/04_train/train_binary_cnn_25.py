from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("data/processed_25_binary")
IMG_SIZE = (25, 25)
BATCH = 64

def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        shuffle=True
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        shuffle=False
    )

    print("Class order:", train_ds.class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25, 25, 3)),
        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

    loss, acc = model.evaluate(test_ds)
    print("\nTest accuracy:", acc)

    out = Path("outputs/models/stop_vs_yield_cnn_25x25.keras")
    out.parent.mkdir(exist_ok=True)
    model.save(out)
    print("Saved:", out)

if __name__ == "__main__":
    main()

