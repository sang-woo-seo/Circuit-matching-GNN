import utils
from trainer import Model_Trainer
import os

ROOT_DIR = os.path.dirname(os.getcwd())


def main(args):
    mt = Model_Trainer(args)
    if args.train_embedder:
        mt.train_embedder()
        exit()
    mt.load_embedder()
    # mt.feature_matching_bb()
    # mt.test_embedder_research(perform_analysis=True)
    mt.test_bb()
    #TODO: mt graph matching

if __name__ == '__main__':
    args = utils.parse_args()
    args.root = ROOT_DIR
    main(args)