class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def __call__(self, score: float) -> bool:
        """Returns True if training should stop early."""
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False
