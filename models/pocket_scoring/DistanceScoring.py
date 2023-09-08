import os

import torch
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from torch_geometric.data import Batch


class DistanceScoring:
    """
    This class scores pockets according to the distance to the true ligand center.
    In a real inference scenario, the true position of the ligand is not available.
    This class is only meant to be used for code testing.
    """

    def __call__(self, batch: Batch) -> list[float]:
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
            results.append(distance.item())
        return results
