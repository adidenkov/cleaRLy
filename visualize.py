from visdom import Visdom
import numpy as np

viz = Visdom()

log_num = 0

def update_viz(reward, log_eps_iter):
    global log_num
    log_num+=1

    viz.line(
        X=np.array([log_num * log_eps_iter]),
        Y=np.array([reward]),
        win="Policy Gradient",
        update='append',
        opts=dict(
            title="Policy Gradient",
            xlabel="episodes",
            ylabel="reward"
        )
    )