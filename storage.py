import torch

class TrajectoryStorage():
    def __init__(self, *args):
        super().__init__()

        for name in args:
            setattr(self, name, "yeet")


    def insert(self, **kwargs):
        to_insert = dict()
        for name, attribute in kwargs.items(): 
            if type(attribute) == torch.Tensor:
                self.trajectories.appened(attribute)
            else:
                self.trajectories.appened(torch.tensor(attribute))
                

    def get_rollout(self):
        pass


    def clear(self):
        del self.trajectories[:]
    