import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _find_multiple(a, b):
    return (-(a // -b)) * b


def trunc_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
):
    # This function is a PyTorch version of jax truncated normal init
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def rms_norm(x, variance_epsilon=1e-5):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return x.to(input_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5)
            )
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion=4.0):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TRMBlock(nn.Module):
    """TRM의 기본 블록
    간단한 Residual MLP Block으로 구성.
    """

    def __init__(self, hidden_size, expansion=4.0):
        super().__init__()
        self.norm_eps = 1e-5
        self.mlp = SwiGLU(hidden_size, expansion)

    def forward(self, x):
        out = self.mlp(x)
        x = rms_norm(x + out, variance_epsilon=self.norm_eps)
        return x


class TRMModel(nn.Module):
    def __init__(self, hidden_size=512, expansion=4.0, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = 768 * 2  # concat CLIP@336
        self.projection = CastedLinear(self.input_dim, hidden_size, bias=True)

        # z: Latent Reasoning, y: Answer State
        self.H_init = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)  # y_init
        self.L_init = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)  # z_init

        # Reasoning Network
        self.layers = nn.ModuleList(
            [TRMBlock(hidden_size, expansion) for _ in range(num_layers)]
        )

        self.lm_head = CastedLinear(hidden_size, 1, bias=False)  # Binary Classification
        # self.q_head = CastedLinear(hidden_size, 2, bias=True)  # Halting Head

        self.H_init = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.L_init = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Q-Head(Halting) 특수 초기화
        # with torch.no_grad():
        # self.q_head.weight.zero_()
        # self.q_head.bias.fill_(-5.0)

    def forward_net(self, state, context):
        # state: 현재 업데이트 하려는 상태 (z 또는 y)
        # context: 입력으로 들어오는 정보 (x+y 또는 z)
        x = state + context.clone()
        for layer in self.layers:
            x = layer(x)
        return x

    def latent_recursion(self, x, y, z, n=6):
        """
        Inner loop: Updates z (latent) n times, then updates y (answer) once.
        """
        for _ in range(n):
            z = self.forward_net(state=z, context=y + x)

        y = self.forward_net(state=y, context=z)
        return y, z

    def deep_recursion(self, x, y, z, n=6, T=3):
        """
        Outer loop: Runs latent recursion T times.
        Only the last iteration tracks gradients.
        """
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)

        y, z = self.latent_recursion(x, y, z, n)

        logits = self.lm_head(y)  # [B, 1, 1]
        # q_logits = self.q_head(y)  # [B, 1, 2]

        return (y, z), logits  # , q_logits
