import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, fingerprint_model, graphs_model, hidden_dim, out_dim):
        super(FusionModel, self).__init__()
        self.fingerprint_model = fingerprint_model
        self.graphs_model = graphs_model
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fingerprint_dim = 128

        self.attention_fusion = None
        self.fc = None

    def forward(self,atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask,x, edge_index, edge_weight, batch, indices=None):

        _, _, fingerprint_feature = self.fingerprint_model(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask,indices=indices)

        _, _, graph_feature = self.graphs_model(x, edge_index, batch)

        if self.attention_fusion is None:
            graph_dim = graph_feature.shape[-1]
            fused_dim = self.fingerprint_dim + graph_dim
            self.attention_fusion = MultiHeadAttentionFusion(
                fingerprint_dim=self.fingerprint_dim,
                graph_dim=graph_dim,
                fused_dim=fused_dim,
                num_heads=4
            ).to(graph_feature.device)
            self.task_heads = nn.ModuleDict({
                task: nn.Linear(fused_dim, 2) for task in ['BS', 'RT', 'FHM', 'SHM']
            })
            self.task_heads = self.task_heads.to(graph_feature.device)

        fused_feature = self.attention_fusion(fingerprint_feature, graph_feature)  # [batch_size, fused_dim]
        output_list = []
        for task in ['BS', 'RT', 'FHM', 'SHM']:
            output_list.append(self.task_heads[task](fused_feature))
        output = torch.cat(output_list, dim=1)

        return output

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, fingerprint_dim, graph_dim, fused_dim, num_heads=4):
        super(MultiHeadAttentionFusion, self).__init__()
        self.fingerprint_dim = fingerprint_dim
        self.graph_dim = graph_dim
        self.fused_dim = fused_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim)
        )

    def forward(self, fingerprint_feature, graph_feature):
        combined = torch.cat([fingerprint_feature, graph_feature], dim=-1)  # [batch_size, fused_dim]
        combined = combined.unsqueeze(1)  # [batch_size, 1, fused_dim]
        attn_output, _ = self.multihead_attn(combined, combined, combined)  # [batch_size, 1, fused_dim]
        attn_output = attn_output.squeeze(1)  # [batch_size, fused_dim]
        fused_feature = self.mlp(attn_output)  # [batch_size, fused_dim]
        return fused_feature
