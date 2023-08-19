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

    # general arguments
    argument_parser.add_argument("--logging_level", default="info", choices=("debug", "info", "warning", "error"), help="Level of verbosity in logging.", type=str)

    # p2rank arguments
    argument_parser.add_argument("--p2rank_path", default="/home/diego/Universidad/Harvard/Lab/methods/p2rank_2.4.1/prank", help="Path to the p2rank executable.", type=str)
    argument_parser.add_argument("--ignore_cache", default=False, help="If this flag is set, the pockets cache will be ignored and p2rank will be forced to run.", action="store_true")

    # pocket scoring arguments
    argument_parser.add_argument("--smina_path", default="/home/diego/Universidad/Harvard/Lab/methods/smina.static", help="Path to the SMINA executable", type=str)
    argument_parser.add_argument("--scoring_batch_size", default=1, help="Batch size for the scoring model.", type=int)

    # pocket docking arguments
    argument_parser.add_argument("--top_k", default=1, help="How many of the top ranked pockets to dock to", type=int)
    argument_parser.add_argument("--docking_batch_size", default=1, help="Batch size for the docking model.", type=int)

    return argument_parser.parse_args()


if __name__ == '__main__':
    logging_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    namespace = parse_arguments()
    logging.basicConfig(level=logging_map[namespace.logging_level],
                        format="%(asctime)s: %(levelname)s | %(message)s")

    pipeline = TwoStepBlindDocking(
        p2rank_executable_path=namespace.p2rank_path,
        pocket_scoring_module=DebugScoring(mode="p2rank"),
        pocket_docking_module=SMINADocking(smina_path=namespace.smina_path),
        top_k=namespace.top_k,
        scoring_batch_size=namespace.scoring_batch_size,
        docking_batch_size=namespace.docking_batch_size,
        ignore_cache=namespace.ignore_cache
    )

    data_path = "data/PDBBind_processed"
    pdbs = []
    for pdb in tqdm(os.listdir(data_path)[:10]):
        pdbs.append(ProteinLigandComplex(name=pdb, protein_path=os.path.join(data_path, pdb, f"{pdb}_protein_processed.pdb"),
                                         ligand_path=os.path.join(data_path, pdb, f"{pdb}_ligand.mol2")))

    print(pipeline.evaluate(pdbs))
