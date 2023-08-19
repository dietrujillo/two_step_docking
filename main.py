import argparse
import logging
import os

from tqdm import tqdm

from dataloader.protein_ligand_complex import ProteinLigandComplex
from models.pocket_docking.SMINADocking import SMINADocking
from models.pocket_scoring.DebugScoring import DebugScoring
from pipeline import TwoStepBlindDocking


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser()

    # ARGUMENTS HERE

    return argument_parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    namespace = parse_arguments()

    pipeline = TwoStepBlindDocking(
        p2rank_executable_path="/home/diego/Universidad/Harvard/Lab/methods/p2rank_2.4.1/prank",
        pocket_scoring_module=DebugScoring(mode="p2rank"),
        pocket_docking_module=SMINADocking(smina_path="/home/diego/Universidad/Harvard/Lab/methods/smina.static"),
        top_k=1,
        scoring_batch_size=1,
        docking_batch_size=1,
        use_cached_pockets=True
    )

    data_path = "data/PDBBind_processed"
    pdbs = []
    for pdb in tqdm(os.listdir(data_path)[:10]):
        pdbs.append(ProteinLigandComplex(name=pdb, protein_path=os.path.join(data_path, pdb, f"{pdb}_protein_processed.pdb"),
                                         ligand_path=os.path.join(data_path, pdb, f"{pdb}_ligand.mol2")))

    print(pipeline.evaluate(pdbs))

