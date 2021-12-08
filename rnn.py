
import numpy as np
from numpy.lib.utils import safe_eval


class RecurrentLayer(object):
    def __init__(
        self, input_width, state_width,
        activator, learning_rate
        ):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros(
            (state_width, 1)
        ))
        self.U = np.random.uniform(
            -1e-4, 1e4, 
            (state_width, 1)
            )
        
        self.W = np.random.uniform(
            -1e-4, 1e-4,
            (state_width, input_width)
            )
    
    def forward(self, input_array):
        self.times += 1
        state = (
            np.dot(self.U, input_array)
            + np.dot(self.W, self.state_list[-1])
            )
        element_wise_op(state, self.activator.forward)