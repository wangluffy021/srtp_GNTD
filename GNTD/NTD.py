import torch
from torch.nn import Embedding, Linear, PReLU, Parameter

class NTD(torch.nn.Module):
    def __init__(self, n_x, n_y, n_g, rank, random_state=1234567):
        super().__init__()
        torch.manual_seed(random_state)

        self.rank_g, self.rank_x, self.rank_y = self._parse_tucker_rank(rank)

        # Define embedding layer along x, y, g modes
        self.embedding_x = Embedding(n_x, self.rank_x)
        self.embedding_y = Embedding(n_y, self.rank_y)
        self.embedding_g = Embedding(n_g, self.rank_g)
        # Define nonlinear mapping layer along x, y, g modes
        self.lin_x_1 = Linear(self.rank_x, self.rank_x)
        self.lin_y_1 = Linear(self.rank_y, self.rank_y)
        self.lin_g_1 = Linear(self.rank_g, self.rank_g)
        self.prelu = PReLU(init=0.9)
        self.core = Parameter(torch.empty(self.rank_g, self.rank_x, self.rank_y))
        torch.nn.init.xavier_uniform_(self.core.reshape(self.rank_g, -1))

    @staticmethod
    def _parse_tucker_rank(rank):
        if isinstance(rank, int):
            if rank <= 0:
                raise ValueError("rank must be a positive integer")
            return rank, rank, rank

        if not isinstance(rank, (list, tuple)) or len(rank) != 3:
            raise ValueError("rank must be an int or a tuple/list with three Tucker ranks (L, M, N)")

        rank_g, rank_x, rank_y = tuple(int(r) for r in rank)
        if min(rank_g, rank_x, rank_y) <= 0:
            raise ValueError("all Tucker ranks must be positive integers")

        return rank_g, rank_x, rank_y

    def forward(self, x_index, y_index, g_index):

        # Linear factors
        x = self.embedding_x(x_index)
        y = self.embedding_y(y_index)
        g = self.embedding_g(g_index)

        # Nonlinear factors
        x = self.lin_x_1(x)
        x = self.prelu(x)
        y = self.lin_y_1(y)
        y = self.prelu(y)
        g = self.lin_g_1(g)
        g = self.prelu(g)

        # Tucker-style spatial basis used by graph regularization.
        spatial_basis = torch.einsum('abc,jb,kc->jka', self.core, x, y)
        spatial_basis = spatial_basis.reshape(-1, self.rank_g)

        # Nonlinear aggregation with Tucker core tensor.
        o = torch.einsum('abc,ia,jb,kc->ijk', self.core, g, x, y)
        o = o.relu_()

        return x, y, g, spatial_basis, o
