import mindspore as ms
from mindspore import nn, Model
from mindspore.train.callback import Callback, LossMonitor, ModelCheckpoint, CheckpointConfig
import numpy as np

# -------------------------------------------------
# Callback Recall (classe 1)
# -------------------------------------------------
class ValidationRecall(Callback):
    def __init__(self, model, val_dataset, threshold=0.5):
        self.model = model
        self.val_dataset = val_dataset
        self.threshold = threshold

    def epoch_end(self, run_context):
        tp, fn = 0, 0

        for data in self.val_dataset.create_dict_iterator():
            x = data["features"]
            y = data["label"]

            logits = self.model.predict(x)
            logits_np = np.clip(logits.asnumpy(), -30, 30)
            probs = 1 / (1 + np.exp(-logits_np))

            preds = (probs >= self.threshold).astype(np.int32)
            y_true = y.asnumpy().astype(np.int32)

            tp += np.sum((preds == 1) & (y_true == 1))
            fn += np.sum((preds == 0) & (y_true == 1))

        recall = tp / (tp + fn + 1e-8)
        print(f"[Validation] recall = {recall:.4f}")


# -------------------------------------------------
# Entraînement
# -------------------------------------------------
def train_model(train_ds, val_ds, input_dim, pos_weight, epochs=30):
    from model.model import AlcoholMLP

    net = AlcoholMLP(input_dim)

    loss = nn.BCEWithLogitsLoss(
        pos_weight=ms.Tensor(pos_weight, ms.float32)
    )

    optimizer = nn.Adam(
        net.trainable_params(),
        learning_rate=1e-3
    )

    model = Model(
        net,
        loss_fn=loss,
        optimizer=optimizer
    )

    # Callbacks
    loss_monitor = LossMonitor(per_print_times=100)

    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=train_ds.get_dataset_size(),
        keep_checkpoint_max=5
    )

    ckpt_cb = ModelCheckpoint(
        prefix="alcohol_mlp",
        directory="checkpoints",
        config=ckpt_config
    )

    val_recall_cb = ValidationRecall(
        model=model,
        val_dataset=val_ds,
        threshold=0.4
    )

    # Train
    model.train(
        epochs,
        train_ds,
        callbacks=[loss_monitor, ckpt_cb, val_recall_cb],
        dataset_sink_mode=False
    )

    return model
