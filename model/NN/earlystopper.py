#early stopper class
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_loss = float('inf')
        self.counter = 0

    def early_stop(self, loss_value:float) -> bool:
        if loss_value < self.best_loss - self.min_delta: #improvement
            self.best_loss = loss_value
            self.counter = 0
        else:
            self.counter += 1
            return self.counter >= self.patience

