from visdom import Visdom
import numpy as np

viz = Visdom()


def update_viz(episode, reward):

    viz.line(
        X=np.array([episode]),
        Y=np.array([reward]),
        win="Policy Gradient",
        update='append',
        opts=dict(
            title="Policy Gradient",
            xlabel="episodes",
            ylabel="reward"
        )
    )