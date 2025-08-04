import torch
import torch.nn as nn
import math
from typing import List, Dict

class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts for multi-task learning with scenario-aware gating."""

    def __init__(
        self,
        num_features: int,
        num_fields: int,
        embed_dim: int,
        experts_hidden: int,
        num_experts: int,
        task_hidden: int,
        tasks: List[str],
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_features, embed_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_fields * embed_dim, experts_hidden),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Linear(num_fields * embed_dim, num_experts),
                    nn.Softmax(dim=-1),
                )
                for t in tasks
            }
        )
        self.towers = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Linear(experts_hidden, task_hidden),
                    nn.ReLU(),
                    nn.Linear(task_hidden, 1),
                )
                for t in tasks
            }
        )
        self.tasks = tasks

    def forward(self, x: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch, fields)
        embed = self.embedding(x)  # (batch, fields, embed_dim)
        flat = embed.view(embed.size(0), -1)  # (batch, fields * embed_dim)

        expert_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)  # (batch, num_experts, hidden)

        outputs = {}
        for t in self.tasks:
            gate_weights = self.gates[t](flat).unsqueeze(-1)  # (batch, num_experts, 1)
            mixed = torch.sum(gate_weights * expert_outputs, dim=1)  # (batch, hidden)
            logit = self.towers[t](mixed).squeeze(-1)
            outputs[t] = torch.sigmoid(logit)
        return outputs