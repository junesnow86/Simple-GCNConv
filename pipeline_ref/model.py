import torch


class RNNModel(torch.nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_class=5, vocab_size=30522, depth=1, dropout=True, dropout_p=0.2,
                 normalize_mode='bn', res=True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.res = res

        if normalize_mode == 'bn':
            self.normalize = torch.nn.BatchNorm1d(hidden_dim)
        elif normalize_mode == 'ln':
            self.normalize = torch.nn.LayerNorm(hidden_dim)
        elif normalize_mode == 'in':
            self.normalize = torch.nn.InstanceNorm1d(hidden_dim)

        self.embedding = torch.nn.Embedding(vocab_size+1, input_dim)
        self.GRUCells = torch.nn.ModuleList()
        self.GRUCells.append(torch.nn.GRUCell(input_dim, hidden_dim))
        for _ in range(depth-1):
            self.GRUCells.append(torch.nn.GRUCell(hidden_dim, hidden_dim))

        self.dropout_layer = torch.nn.Dropout(dropout_p)
        self.fc = torch.nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, input_dim = x.shape

        h = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.depth)]
        for t in range(seq_len):
            for i in range(self.depth):
                if i == 0:
                    h[i] = self.GRUCells[i](x[:, t, :], h[i])
                elif i == 1:
                    cell_input = h[i-1] + x[:, t, :] if self.res else h[i-1]
                    h[i] = self.GRUCells[i](cell_input, h[i])
                else:
                    cell_input = h[i-1] + h[i-2] if self.res else h[i-1]
                    h[i] = self.GRUCells[i](cell_input, h[i])
            for k, hk in enumerate(h):
                if self.dropout:
                    h[k] = self.dropout_layer(hk)
                h[k] = self.normalize(hk)

        out = self.fc(h[-1])
        return out
