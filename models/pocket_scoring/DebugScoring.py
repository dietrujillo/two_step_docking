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
        if self.mode not in {"distance", "p2rank"}:
            raise ValueError(f"The parameter \"mode\" must be one of {{\"distance\", \"p2rank\"}}. Got {mode}.")

    def call_distance(self, batch: Batch):
        """
        Compute a score based on the distance from the true ligand position to the center of the pocket.
        In order to do this, the true position of the ligand must be known; and as such,
        the function is meant to be used for testing.
        :param batch: data batch of HeteroData objects.
        :return: the score for each element in the batch as the distance from the ligand to the pocket centroid.
        """
        results = []
        batch = batch.to_data_list()
        for item in batch:
            if item["rdkit_reference_ligand"] != {}:
                reference_ligand = item["rdkit_reference_ligand"]
            else:
                reference_ligand = item["rdkit_ligand"]
            ligand_center = ComputeCentroid(reference_ligand.GetConformer())
            ligand_center = torch.Tensor([ligand_center.x, ligand_center.y, ligand_center.z])
            distance = torch.norm(ligand_center - item["pocket_centroid"])
            results.append(distance)
        return results

    def call_p2rank(self, batch: Batch):
        """
        Rank pockets based on their p2rank score.
        This means we skip any ligand-dependent ranking and keep the p2rank predictions.
        :param batch: data batch of HeteroData objects.
        :return: pocket score based on the p2rank prediction.
        """
        results = []
        for protein_path in batch["protein_path"]:
            results.append(int(os.path.basename(protein_path).split("_")[1][:-4]))
        return results

    def __call__(self, batch: Batch):
        """
        Score a batch of pocket-ligand pairs according to how well they bind together.
        Lower score means higher likelihood.
        :param batch: data batch of HeteroData objects.
        :return: a list of scores for each element in the batch.
        """
        if self.mode == "distance":
            return self.call_distance(batch)
        elif self.mode == "p2rank":
            return self.call_p2rank(batch)
