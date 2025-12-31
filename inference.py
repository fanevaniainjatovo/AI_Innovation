import mindspore as ms
from mindspore import Model
import mindspore.ops as ops
from model.dataset import create_dataset
from model.model import AlcoholMLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

TEST_CSV = "data/test.csv"
CKPT_PATH = "checkpoints/alcohol_mlp_1-10_556.ckpt"

test_ds = create_dataset(TEST_CSV)

input_dim = next(test_ds.create_dict_iterator())["features"].shape[1]

net = AlcoholMLP(input_dim)
ms.load_checkpoint(CKPT_PATH, net)
net.set_train(False)

model = Model(net)

y_true, y_pred = [], []

for batch in test_ds.create_dict_iterator():
    logits = model.predict(batch["features"])
    probs = ops.Sigmoid()(logits).asnumpy()
    preds = (probs >= 0.5).astype(int)

    y_pred.extend(preds)
    y_true.extend(batch["label"].asnumpy())

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1       :", f1_score(y_true, y_pred))
