from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("data/processed_25")
IMG_SIZE = (25, 25)
BATCH = 16  # small dataset -> small batch is fine

def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",   # 0,1,2
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",
        shuffle=True
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Class order:", class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(25, 25, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
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
    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

    loss, acc = model.evaluate(test_ds)
    print("\nTest accuracy:", acc)

    out_path = Path("outputs/models/stop_yield_other_mlp_25x25.keras")
    out_path.parent.mkdir(exist_ok=True)
    model.save(out_path)
    print("Saved model to:", out_path)

if __name__ == "__main__":
    main()
