import numpy as np


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model=None, path=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 保存模型权重
        model.save_weights(f"{path}/checkpoint.ckpt")
        model.save_weights(f"{path}/checkpoint.h5")
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    print_lr = True
    learning_rate = args.learning_rate
    # 计算新的学习率
    exponent = (epoch - 1) // 1
    lr = learning_rate * (0.5 ** exponent)
    # 更新优化器学习率
    optimizer.learning_rate.assign(lr)
    if print_lr:
        print(f'Updating learning rate to {lr}')
    return lr