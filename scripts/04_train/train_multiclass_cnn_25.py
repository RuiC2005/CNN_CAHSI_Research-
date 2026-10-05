from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_DIR = Path("data/processed_25_multiclass")  # built by build_multiclass_dataset_combined.py
IMG_SIZE = (25, 25)
BATCH = 64

def main():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing: {train_dir}. Run the build script first.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=True
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Class order:", class_names)
    print("Num classes:", num_classes)

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

        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=cb
    )

    # Accuracy plot (clean + poster-ready)
    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/models/accuracy_plot.png", dpi=300)
    plt.show()


    # Loss plot (keep yours, just match style)
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/models/loss_plot.png", dpi=300)
    plt.show()


    # Evaluation (leave this as is)
    loss, acc = model.evaluate(test_ds)
    print("Test accuracy:", acc)

    out = Path("outputs/models/gtsrb_multiclass_cnn_25x25.keras")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print("Saved:", out)

if __name__ == "__main__":
    main()