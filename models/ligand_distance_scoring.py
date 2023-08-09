import os

import numpy as np
import pandas as pd
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from torch_geometric.data import Batch


class LigandDistanceScoring:
    """
    This class scores pockets according to the distance to the true ligand center.
    In a real inference scenario, the true position of the ligand is not available.
    This class is only meant to be used for code testing.
    """

    def __init__(self, p2rank_output_folder: str = ".p2rank_cache/p2rank_output") -> None:
        super().__init__()
        self.p2rank_output_folder = p2rank_output_folder

    def __call__(self, batch: Batch, ignore_hydrogen: bool = True):
        results = []
        for protein_path, ligand in zip(batch["protein_path"], batch["rdkit_ligand"]):

            ligand_center = ComputeCentroid(ligand.GetConformer())
            ligand_center = np.array([ligand_center.x, ligand_center.y, ligand_center.z])

            p2rank_predictions = pd.read_csv(
                os.path.join(self.p2rank_output_folder,
                             f"{os.path.basename(protein_path).split('_')[0]}_protein_processed.pdb_predictions.csv"))
            p2rank_predictions["distance_to_ligand"] = p2rank_predictions.apply(
                lambda row: np.linalg.norm(
                    ligand_center - np.array([row["   center_x"], row["   center_y"], row["   center_z"]])), axis=1
            )
            p2rank_predictions["score"] = p2rank_predictions["distance_to_ligand"].rank(ascending=True)
            results.extend(p2rank_predictions["score"].to_list())
        return results
