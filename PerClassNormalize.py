class PerClassNormalize:
    def __init__(self, means, stds):
        self.means = means
        self.stds  = stds

    def __call__(self, spec, label):
        # spec: torch.Tensor, label: integer class index
        mu  = self.means[label]
        std = self.stds[label]
        return (spec - mu) / std
