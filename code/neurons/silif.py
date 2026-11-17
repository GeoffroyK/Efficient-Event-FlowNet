'''
Spikingjelly implementation of the 
SiLIF neuron model from "SiLIF: A Spiking Leaky Integrate-and-Fire Neuron with Learnable Decay and Adaptation" 
Available here: https://arxiv.org/abs/2211.13227.

@GeoffroyK, 21/04/2026
'''

import torch
import torch.nn as nn
from typing import Callable, Optional
from spikingjelly.activation_based.neuron import BaseNode
from spikingjelly.activation_based.neuron import surrogate

class SiLIFNode(BaseNode):
    def __init__(self,
                 hidden_size: int,
                 lambda_alpha_min: float = 10.0,
                 lambda_alpha_max: float = 250.0,
                 lambda_beta_min: float = 2.5,
                 lambda_beta_max: float = 10.0,
                 a_min: float = 0.0,
                 a_max: float = 1.0,
                 b_min: float = 0.0,
                 b_max: float = 2.0,   
                 dt_init: float = 0.004,
                 v_threshold: float = 1.0,
                 v_reset: Optional[float] = 0.0,
                 surrogate_function: Callable = surrogate.ATan(),
                 detach_reset: bool = True,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False,
    ):
        
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq
        )
        
        self.hidden_size = hidden_size
        self.lambda_alpha_min = lambda_alpha_min
        self.lambda_alpha_max = lambda_alpha_max
        self.lambda_beta_min = lambda_beta_min
        self.lambda_beta_max = lambda_beta_max
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.dt_init = dt_init
        
        self._init_learnable_parameters()

        # Adaptation state and previous spike, reset between sequences by spikingjelly
        self.register_memory('w', 0.)
        self.register_memory('s_prev', 0.)

    def _init_learnable_parameters(self):
        log_alpha_min = torch.log(torch.tensor(self.lambda_alpha_min + 1e-6))
        log_alpha_max = torch.log(torch.tensor(self.lambda_alpha_max + 1e-6))
        log_beta_min = torch.log(torch.tensor(self.lambda_beta_min + 1e-6))
        log_beta_max = torch.log(torch.tensor(self.lambda_beta_max + 1e-6))
        
        # Logarithmic (re)parametrization
        self.lambda_log_alpha = nn.Parameter(torch.empty((self.hidden_size,)))
        nn.init.uniform_(self.lambda_log_alpha, a=log_alpha_min, b=log_alpha_max)
        
        self.lambda_log_beta = nn.Parameter(torch.empty((self.hidden_size,)))
        nn.init.uniform_(self.lambda_log_beta, a=log_beta_min, b=log_beta_max)
        
        # Learnable time step
        self.dt = nn.Parameter(torch.empty((self.hidden_size,)))
        nn.init.constant_(self.dt, torch.log(torch.tensor(self.dt_init)))
        
        # Heterogeneous parameters
        self.a = nn.Parameter(torch.empty((self.hidden_size,)))
        nn.init.uniform_(self.a, a=self.a_min, b=self.a_max)
      
        self.b = nn.Parameter(torch.empty((self.hidden_size,)))
        nn.init.uniform_(self.b, a=self.b_min, b=self.b_max)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v_float_to_tensor(x)

        # Recover actual decay factors from log-space parameters
        alpha = torch.exp(-torch.exp(self.lambda_log_alpha) * torch.exp(self.dt))
        beta  = torch.exp(-torch.exp(self.lambda_log_beta)  * torch.exp(self.dt))

        # Clamp a and b to their valid ranges (Algorithm 1)
        a = self.a.clamp(self.a_min, self.a_max)
        b = self.b.clamp(self.b_min, self.b_max)

        # w ← β·w + a·u + b·s  (s_prev is spike from previous timestep)
        self.w = beta * self.w + a * self.v + b * self.s_prev

        # u ← α·(u - s) + (1-α)·(I_t - w)
        self.v = alpha * (self.v - self.s_prev) + (1 - alpha) * (x - self.w)

        # s ← Θ(u - θ)  using surrogate gradient
        spike = self.neuronal_fire()

        # store spike for next timestep's w update
        self.s_prev = spike.detach() if self.detach_reset else spike

        return spike

    @property
    def A(self):
        '''
        State transition matrix (2x2) per neuron (heterogeneous).
        Shape: (hidden_size, 2, 2)
        '''
        alpha = torch.exp(-torch.exp(self.lambda_log_alpha) * torch.exp(self.dt))
        beta = torch.exp(-torch.exp(self.lambda_log_beta) * torch.exp(self.dt))
        a = self.a.clamp(self.a_min, self.a_max)

        A = torch.zeros(self.hidden_size, 2, 2, device=alpha.device)
        A[:, 0, 0] = alpha - (1 - alpha) * a   # v <- v coefficient
        A[:, 0, 1] = -(1 - alpha) * beta        # v <- w coefficient
        A[:, 1, 0] = a                           # w <- v coefficient
        A[:, 1, 1] = beta                        # w <- w coefficient
        return A
    
    @property
    def B(self) -> torch.Tensor:
        '''
        Input matrix (2x1) per neuron (heterogeneous).
        Shape: (hidden_size, 2)
        '''
        alpha = torch.exp(-torch.exp(self.lambda_log_alpha) * torch.exp(self.dt))
        B = torch.zeros(self.hidden_size, 2, device=alpha.device)
        B[:, 0] = 1 - alpha
        B[:, 1] = 0
        return B