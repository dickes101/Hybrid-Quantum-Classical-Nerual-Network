import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from sklearn.metrics import accuracy_score

from QGRNN.data import Dataset
from QGRNN.model import ImprovedHybridQGNN
from QGRNN.train_wrapper import TrainWrapper
from QGRNN.utils import TrainingHistory, config


def evaluate(model, dataset, vq_idx, loss_fn=None):
    
    model.set_train(False)
    logits, _ = model(dataset.features, dataset.adj, vq_indices=vq_idx)
    preds = ops.Argmax(axis=1)(logits)
    y = dataset.labels.asnumpy()
    p = preds.asnumpy()

    train_acc = accuracy_score(
        y[dataset.train_mask.asnumpy()],
        p[dataset.train_mask.asnumpy()]
    )
    val_acc = accuracy_score(
        y[dataset.val_mask.asnumpy()],
        p[dataset.val_mask.asnumpy()]
    )
    test_acc = accuracy_score(
        y[dataset.test_mask.asnumpy()],
        p[dataset.test_mask.asnumpy()]
    )

    if loss_fn is not None and dataset.val_mask.asnumpy().sum() > 0:
        val_idx = np.where(dataset.val_mask.asnumpy())[0]
        val_idx = Tensor(val_idx, ms.int32)
        val_logits = ops.gather(logits, val_idx, 0)
        val_labels = ops.gather(dataset.labels, val_idx, 0)
        val_loss = float(loss_fn(val_logits, val_labels).asnumpy())
    else:
        val_loss = 0.0

    return train_acc, val_acc, test_acc, val_loss

def train_improved_model(config):

    dataset = Dataset(dataset=config['dataset'], k=config['k'], extra_num=config['extra_num'])

    num_classes = len(np.unique(dataset.labels.asnumpy()))
    edge_index = Tensor(dataset.edge_index, ms.int32)
    vq_idx = dataset.vq_indices_ms

    model = ImprovedHybridQGNN(
        in_feats=dataset.features.shape[1],
        hidden_size=config['hidden_dim'],
        num_classes=num_classes,
        num_qubits=config['num_qubits'],
        quantum_depth=config['quantum_depth'],
    )

    params = model.trainable_params()
    optimizer = nn.Adam(params, 
                        learning_rate=config['learning_rate'], 
                        weight_decay=config['weight_decay'])
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    train_net = TrainWrapper(model, loss_fn, edge_index,
                             lambda_reg=config['edge_reg'])

    history = TrainingHistory()
    best = {'val': 0, 'params': None, 'epoch': 0}
    patience = 0

    print("Start...")
    print(config)

    for epoch in range(config['epochs']):
        model.set_train(True)

        total_loss, cls_loss, edge_loss = train_net(
            dataset.features, dataset.adj, dataset.labels, dataset.train_mask, vq_indices=vq_idx
        )

        grads = ms.grad(train_net, grad_position=None, weights=params)(
            dataset.features, dataset.adj, dataset.labels, dataset.train_mask, vq_idx
        )
        optimizer(grads)

        if epoch % config['eval_interval'] == 0:

            train_acc, val_acc, test_acc, val_loss = evaluate(model, dataset, vq_idx, loss_fn=loss_fn)
            history.update(epoch, cls_loss.asnumpy().item(), val_loss, train_acc, val_acc, test_acc)

            print(f"[{epoch:03d}] loss={float(total_loss):.4f}, "
                  f"train={train_acc:.4f}, val={val_acc:.4f}, test={test_acc:.4f}, "
                  f"val_loss={val_loss:.4f}")

            if val_acc > best['val']:
                best.update({
                    'val': val_acc,
                    'params': [p.value().asnumpy() for p in params],
                    'epoch': epoch
                })
                patience = 0
            else:
                patience += 1
                if patience >= config['patience']:
                    print(f"Early stop at epoch {epoch}")
                    break

    if best['params'] is not None:
        for i, p in enumerate(params):
            p.set_data(Tensor(best['params'][i], ms.float32))
        print(f"Best Model（epoch {best['epoch']}）")

    _, _, final_test_acc, _ = evaluate(model, dataset, vq_idx)
    print(f"Final test: {final_test_acc:.4f}")

    return model, final_test_acc, history, dataset

ms.set_context(device_target="GPU", device_id=7)
ms.set_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    model, test_acc, history, dataset = train_improved_model(config)
    history.plot("training_curve.png")