import numpy as np
import torch
from torch import nn


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """
    Cosine annealing with restarts.
    This is decribed in the paper https://arxiv.org/abs/1608.03983.
    Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``
    t_max : ``int``
        The maximum number of iterations within the first cycle.
    eta_min : ``float``, optional (default=0)
        The minimum learning rate.
    last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.
    factor : ``float``, optional (default=1)
        The factor by which the cycle length (``T_max``) increases after each restart.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        assert t_max > 0
        assert eta_min >= 0
        if t_max == 1 and factor == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                           "rate since T_max = 1 and factor = 1.")
        self.t_max = t_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = t_max
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
                self.eta_min + ((lr - self.eta_min) / 2) * (
                        np.cos(
                                np.pi *
                                (self._cycle_counter % self._updated_cycle_len) /
                                self._updated_cycle_len
                        ) + 1
                )
                for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step

        return lrs


if __name__ == '__main__':
    lin = nn.Linear(128, 256)

    optim = torch.optim.SGD(lin.parameters(), lr=0.01)
    scheduler = CosineWithRestarts(optim, t_max=10, eta_min=0.0001)

    for _ in range(100):
        cur_lr = scheduler.get_lr()
        print(cur_lr[0])
        scheduler.step()
