import torch
from torch_geometric.nn.models import DimeNetPlusPlus

from models.pocket_scoring.ProNet.pronet import ProNet


class AffinityScoring(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protein_embedding = ProNet(out_channels=128)
        self.protein_embedding_head = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU()
        )
        self.ligand_embedding = DimeNetPlusPlus(hidden_channels=128,
                                                out_channels=128,
                                                num_blocks=4,
                                                int_emb_size=64,
                                                basis_emb_size=8,
                                                out_emb_channels=128,
                                                num_spherical=7,
                                                num_radial=6,
                                                cutoff=5.0,
                                                max_num_neighbors=32,
                                                envelope_exponent=5,
                                                num_before_skip=1,
                                                num_after_skip=2,
                                                num_output_layers=3)
        self.ligand_embedding_head = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU()
        )
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(257, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        protein_embedding = self.protein_embedding(inputs)[0]
        protein_embedding = self.protein_embedding_head(protein_embedding)
        ligand_embedding = self.ligand_embedding_head(self.ligand_embedding(inputs["ligand"].x[:, 0],
                                                                            inputs["ligand"].pos.float(),
                                                                            inputs["ligand"].batch))
        embeddings = torch.concat([protein_embedding, ligand_embedding,
                                   inputs["pocket_probability"].unsqueeze(-1).float()], dim=-1)
        out = self.mlp_head(embeddings)
        return out
