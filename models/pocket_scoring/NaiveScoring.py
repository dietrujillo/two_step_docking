import os

from torch_geometric.data import Batch


class NaiveScoring:
    """
    This class scores pockets according to their predicted "ligandability" from p2rank. The method is ligand-agnostic,
     meaning predictions are based on the pocket structure only and are likely to be suboptimal.
    """

    def __call__(self, batch: Batch) -> list[float]:
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
