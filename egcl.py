from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform


class E_GCL_GKN(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, kernel, depth, act_fn=nn.ReLU(), normalize=False,
                 coords_agg='mean',
                 root_weight=True, residual=True, bias=True):
        super().__init__()
        self.in_channels = input_nf
        self.out_channels = output_nf
        self.act_fn = act_fn
        self.kernel = kernel
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.epsilon = 1e-8
        self.depth = depth
        self.residual = residual

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(input_nf, output_nf))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_nf))
        else:
            self.register_parameter('bias', None)

        layer = nn.Linear(2 * hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, 2 * hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.kernel)
        # reset(self.coord_mlp)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def edge_conv(self, source, edge_attr, edge_index):
        row, col = edge_index
        out = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        # out = torch.cat([coord[row], coord[col], out], dim=1)
        weight = self.kernel(out).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(source.unsqueeze(1), weight).squeeze(1)

    def node_conv(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        # agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))

        if self.root is not None:
            agg = agg + torch.mm(x, self.root)
        if self.bias is not None:
            agg = agg + self.bias
        out = self.act_fn(agg) / self.depth
        if self.residual:
            out = x + out
        return out

    def coord_conv(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = agg / self.depth + coord
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return coord_diff

    def forward(self, h, edge_index, coord_curr, edge_attr, node_attr=None):
        row, col = edge_index
        coord_diff = self.coord2radial(edge_index, coord_curr)
        edge_feat = self.edge_conv(h[col], edge_attr, edge_index)
        coord_curr = self.coord_conv(coord_curr, edge_index, coord_diff, edge_feat)
        h = self.node_conv(h, edge_index, edge_feat, node_attr)

        return h, coord_curr


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
