import os

import torch
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from torch_geometric.data import Batch


class DebugScoring:
    """
    This class scores pockets according to the distance to the true ligand center.
    In a real inference scenario, the true position of the ligand is not available.
    This class is only meant to be used for code testing.
    """

    def __init__(self, p2rank_output_folder: str = ".p2rank_cache/p2rank_output", mode="distance"):
        super().__init__()
        self.p2rank_output_folder = p2rank_output_folder
        self.mode = mode
        assert self.mode in {"distance", "p2rank"}

    def call_distance(self, batch: Batch):
        results = []
        for ligand, centroid in zip(batch["rdkit_ligand"], batch["pocket_centroid"]):
            ligand_center = ComputeCentroid(ligand.GetConformer())
            ligand_center = torch.Tensor([ligand_center.x, ligand_center.y, ligand_center.z])

            distance = torch.norm(ligand_center - centroid)
            results.append(distance)
        return results

    def call_p2rank(self, batch: Batch):
        results = []
        for protein_path in batch["protein_path"]:
            results.append(int(os.path.basename(protein_path).split("_")[1][:-4]))
        return results

    def __call__(self, batch: Batch, ignore_hydrogen: bool = True):
        if self.mode == "distance":
            return self.call_distance(batch)
        elif self.mode == "p2rank":
            return self.call_p2rank(batch)
