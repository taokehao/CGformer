from __future__ import print_function, division

import torch
import torch.nn as nn
from torch_geometric.utils import degree


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------
        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------
        atom_in_fea: torch.Tensor shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------
        atom_out_fea: torch.Tensor shape (N, atom_fea_len)
          Atom hidden features after convolution
        """
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)  # (N, M, 2*atom_fea_len + nbr_fea_len)
        total_gated_fea = self.fc_full(total_nbr_fea)  # (N, M, 2*atom_fea_len)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)  # (N, M, 2*atom_fea_len)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # Each (N, M, atom_fea_len)
        nbr_filter = self.sigmoid(nbr_filter)  # (N, M, atom_fea_len)
        nbr_core = self.softplus1(nbr_core)  # (N, M, atom_fea_len)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # (N, atom_fea_len)
        nbr_sumed = self.bn2(nbr_sumed)  # (N, atom_fea_len)
        out = self.softplus2(atom_in_fea + nbr_sumed)  # (N, atom_fea_len)
        return out


# ------------------- Graphormer Components -------------------

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        in_degree = self.decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(),
                                               self.max_in_degree - 1)  # 每个节点的入度
        out_degree = self.decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(),
                                                self.max_out_degree - 1)

        x = x + self.z_in[in_degree] + self.z_out[out_degree]  # 将每个节点度的数值作为索引，挑选z_in或z_out的每行，形成每个节点的嵌入

        return x

    def decrease_to_max_value(self, x, max_value):
        "限制节点度的最大值"
        x = torch.clamp(x, max=max_value)
        return x


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number dimension
        """
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr=None) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        N = x.size(0)
        adjacency = torch.zeros(N, N, device=x.device)
        adjacency[edge_index[0], edge_index[1]] = 1.0

        if ptr is None:
            a = query.mm(key.transpose(0, 1)) / (query.size(-1) ** 0.5)
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=x.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / (query.size(-1) ** 0.5)

        # Apply adjacency mask
        a = a * adjacency + (1 - adjacency) * (-1e6)
        softmax = torch.softmax(a, dim=-1)
        out = softmax.mm(value)
        return out


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        head_outs = []
        for attention_head in self.heads:
            head_out = attention_head(x, edge_index, ptr)
            head_outs.append(head_out)
        concatenated = torch.cat(head_outs, dim=-1)
        out = self.linear(concatenated)
        return out


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, num_heads, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param num_heads: number of attention heads
        :param max_path_distance: (unused in simplified version)
        """
        super().__init__()
        self.attention = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        x_prime = self.attention(self.ln_1(x), edge_index, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        return x_new


class GraphormerEncoder(nn.Module):
    def __init__(self, layers, node_dim, num_heads, max_path_distance):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim, num_heads, max_path_distance)
            for _ in range(layers)
        ])

    def forward(self, x, edge_index, ptr):
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x


# ------------------- Modified CrystalGraphConvNet with Graphormer -------------------

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties, integrated with Graphormer encoder layers.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, graphormer_layers=1,
                 num_heads=4, max_path_distance=5, node_dim=128, edge_dim=128):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        classification: bool
          Whether the task is classification
        graphormer_layers: int
          Number of Graphormer encoder layers
        num_heads: int
          Number of attention heads in Graphormer
        max_path_distance: int
          Maximum path distance for Graphormer
        node_dim: int
          Hidden dimensions of node features in Graphormer
        edge_dim: int
          Hidden dimensions of edge features in Graphormer
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        # Register Graphormer components
        self.centrality_encoding = CentralityEncoding(max_in_degree=10, max_out_degree=10, node_dim=atom_fea_len)
        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads,
            max_path_distance=max_path_distance  # not used in simplified version
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.ReLU()
                                             for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------
        prediction: torch.Tensor shape (N, )
          Predictions for each crystal
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        # Apply Centrality Encoding
        # Reconstruct edge_index correctly
        N = nbr_fea_idx.size(0)  # Number of atoms
        M = nbr_fea_idx.size(1)  # Number of neighbors per atom
        # Construct edge_index as (2, E)
        src = torch.repeat_interleave(torch.arange(N, device=atom_fea.device), M)
        dst = nbr_fea_idx.view(-1)
        edge_index = torch.stack([src, dst], dim=0)  # Shape: (2, N*M)

        # Apply Centrality Encoding
        atom_fea = self.centrality_encoding(atom_fea, edge_index)

        # Construct ptr from crystal_atom_idx
        ptr = [0]
        for idx_map in crystal_atom_idx:
            ptr.append(ptr[-1] + len(idx_map))
        ptr = torch.tensor(ptr, dtype=torch.long, device=atom_fea.device)

        # Graphormer Encoder
        x = self.graphormer_encoder(atom_fea, edge_index, ptr)

        # Continue with pooling
        crys_fea = self.pooling(x, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.size(0)
        # 使用 torch.stack 和 torch.mean 的矢量化操作
        crys_fea = torch.stack([torch.mean(atom_fea[idx_map], dim=0) for idx_map in crystal_atom_idx], dim=0)
        return crys_fea
