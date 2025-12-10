import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

config = {
        'dataset': 'Cora',
        'k': 5,
        'extra_num': 800,
        'hidden_dim': 128,
        'num_qubits': 8,
        'quantum_depth': 2,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'edge_reg': 0.0,
        'epochs': 400,
        'patience': 100,
        'eval_interval': 5,
    }

class TrainingHistory:
    def __init__(self):
        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc, self.test_acc = [], [], []
        self.epochs = []

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, test_acc):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.test_acc.append(test_acc)

    def plot(self, save_path='training_history.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.epochs, self.train_loss, label='Train Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_loss, label='Val Loss', linewidth=2)
        ax1.legend(); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax2.plot(self.epochs, self.train_acc, label='Train Acc', linewidth=2)
        ax2.plot(self.epochs, self.val_acc, label='Val Acc', linewidth=2)
        ax2.plot(self.epochs, self.test_acc, label='Test Acc', linestyle='--', linewidth=2)
        ax2.legend(); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.show()