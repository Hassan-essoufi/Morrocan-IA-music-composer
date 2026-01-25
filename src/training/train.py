import os
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path

from src.datasets.midi_dataset import MidiDataset
from src.models.transformer_decoder import TransformerDecoder
from src.training.train_utils import build_optimizer, masked_sparse_categorical_crossentropy, CustomSchedule

def maestro_train(config):
    """
    Train Transformer model on MAESTRO dataset, mlflow logging.
    """

    tokens_dir = config["data"]["tokens_dir"]
    max_seq_len = config["data"]["max_seq_len"]

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    warmup_steps = config["training"]["warmup_steps"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["patience"]
    checkpoint_dir = config["training"]["checkpoint_dir"]


    maestro_path = Path(tokens_dir) / "maestro"
    train_file = "train.npz"
    val_file = "val.npz"
    checkpoint_maestro = config["training"]["checkpoint_maestro"]

    # Load datasets
    train_dataset = MidiDataset(
        maestro_path, train_file,
        batch_size, max_seq_len,shuffle=True)

    val_dataset = MidiDataset(
        maestro_path, val_file,
        batch_size, max_seq_len, shuffle=False)

    # Load vocabulary
    vocab_size = max_seq_len + 1

    # Build model
    embed_dim = config["model"]["embed_dim"]
    num_heads = config["model"]["n_heads"]
    num_layers = config["model"]["n_layers"]
    ff_dim = config["model"]["ff_dim"]
    dropout = config["model"]["dropout"]

    model = TransformerDecoder(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # Compile
    scheduler = CustomSchedule(embed_dim, warmup_steps)

    optimizer = build_optimizer(scheduler,
                                weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=masked_sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )
    mlflow.set_experiment("training_on_maestro")
    # Training
    with mlflow.start_run(run_name="maestro_training") as run:
        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("ff_dim", ff_dim)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("warmup_steps", warmup_steps)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("patience", patience)

        patience_counter = 0
        best_epoch = 0
        best_weights = None
        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            history = model.fit(train_dataset,
                                validation_data=val_dataset,
                                epochs=1)

            val_loss = history.history["val_loss"][-1]
            train_loss = history.history["loss"][-1]
            val_acc = history.history.get("val_accuracy", [0])[-1]
            train_acc = history.history.get("accuracy", [0])[-1]

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            # Save checkpoint if validation improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_weights = model.get_weights()
                
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                mlflow.log_metric("best_epoch", best_epoch, step=epoch)
            
            else :
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break

        if best_weights:
            model.set_weights(best_weights)
        model.save(checkpoint_maestro)
        mlflow.tensorflow.log_model(model, "maestro_transformer")

    print(f"Maestro model saved to {checkpoint_maestro}")

def train(config):
    """
    Main training function , mlflow logging.
    """

    tokens_dir = config["data"]["tokens_dir"]
    max_seq_len = config["data"]["max_seq_len"]

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    warmup_steps = config["training"]["warmup_steps"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["patience"]

    checkpoint_maestro = config["training"]["checkpoint_maestro"]
    final_model_path = config["training"]["final_model_path"]
    checkpoint_dir = config["training"]["checkpoint_dir"]

    gnawa_path = Path(tokens_dir) / "gnawa"
    train_file = "train.npz"
    val_file = "val.npz"

    # Load dataset
    train_dataset = MidiDataset(gnawa_path, train_file,
                                batch_size, max_seq_len,shuffle=True)

    val_dataset = MidiDataset(gnawa_path, val_file,
                            batch_size, max_seq_len, shuffle=False)

    # Build model
    
    vocab_size = max_seq_len + 1
    embed_dim=config["model"]["embed_dim"]
    num_heads=config["model"]["n_heads"]
    num_layers=config["model"]["n_layers"]
    ff_dim=config["model"]["ff_dim"]
    dropout=config["model"]["dropout"]

    model = tf.keras.models.load_model(
                            checkpoint_maestro,
                            custom_objects={
                        'CustomSchedule': CustomSchedule,
                        'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy},
                        compile=False)

    scheduler = CustomSchedule(embed_dim, warmup_steps)

    optimizer = build_optimizer(scheduler,
                                weight_decay)
    
    model.compile(optimizer=optimizer, 
                  loss=masked_sparse_categorical_crossentropy,
                  metrics=["accuracy"])
    mlflow.set_experiment("training_on_gnawa")
    with mlflow.start_run(run_name="music_transformer_training") as run:
        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("ff_dim", ff_dim)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("warmup_steps", warmup_steps)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("patience", patience)

        patience_counter = 0
        best_epoch = 0
        best_weights  = None
        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            history = model.fit(train_dataset, validation_data=val_dataset,
                                epochs=1)

            val_loss = history.history["val_loss"][-1]
            train_loss = history.history["loss"][-1]
            val_acc = history.history.get("val_accuracy", [0])[-1]
            train_acc = history.history.get("accuracy", [0])[-1]

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            # Save checkpoint if validation improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights =  model.get_weights()
                patience_counter = 0
                
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                mlflow.log_metric("best_epoch", best_epoch, step=epoch)
            
            else :
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break
            
        if best_weights:
            model.set_weights(best_weights)
        # Save Checkpoint
        model.save(final_model_path)
        mlflow.tensorflow.log_model(model, "music_transformer")
    print(f"Final model saved to {final_model_path}")
