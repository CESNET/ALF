import argparse
import logging
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

import alf.anotator
import alf.context_manager
import alf.d_manager
import alf.engine
import alf.evaluator
import alf.input_manager
import alf.ml_model
import alf.postprocess
import alf.preprocess
import alf.query_strategy

ContextProvider = alf.context_manager.ContextProvider
DbProvider = alf.d_manager.DbProvider

logging.basicConfig(
    stream=sys.stdout,
    format='[%(asctime)s]: %(message)s',
    level=logging.DEBUG
)

DATASET_COLUMNS = [
    'bytes_rev',
    'bytes',
    'packets',
    'packets_rev',
    'packets_sum',
    'bytes_ration',
    'num_pkts_ration',
    'time',
    'av_pkt_size',
    'av_pkt_size_rev',
    'var_pkt_size',
    'var_pkt_size_rev',
    'median_pkt_size',
    'median_pkt_size_rev',
    'mindelay',
    'avgdelay',
    'maxdelay',
    'bursts',
    'fizzles',
    'time_leap_ration',
    'autocorr',
    'stSum',
    'ndSum',
    'rdSum'
]

parser = argparse.ArgumentParser(
    description='Alf NEMEA experiment in dry run mode (no database).')
parser.add_argument(
    "--id",
    type=str, help="Experiment ID", required=True)
parser.add_argument(
    "--workdir",
    type=str, help="Working directory", required=True)
parser.add_argument(
    "--model",
    type=str, help="Model name", required=True)
parser.add_argument(
    "--query_strategy",
    type=str, help="Query strategy name", required=True)
parser.add_argument(
    "--blacklist",
    type=str, help="Blacklist of DOH servers file", required=True)
parser.add_argument(
    "--dpath",
    type=str, help="Path to D_0 dataset", required=True)
parser.add_argument(
    "--query_nmax",
    type=int, help="Max number of queried flows", required=False)
parser.add_argument(
    "--query_threshold",
    type=float, help="Threshold for score in query strategy", required=False)
parser.add_argument(
    "--beta",
    type=float, help="Beta for density staregy", required=False)
parser.add_argument(
    "--input",
    type=str, help="Input type (folder or socket", required=True)
parser.add_argument(
    "--input_def",
    type=str, help="Input definition (name of folder or socket", required=True)
parser.add_argument(
    "--postprocessor",
    type=str, help="postprocessor procedure", required=False)
parser.add_argument(
    "--threshold_greedy",
    type=float, help="greedy threshold", required=False)
parser.add_argument(
    "--budget",
    type=float, help="Budget", required=False)
parser.add_argument(
    "--reward",
    type=int, help="Reward", required=False)
parser.add_argument(
    "--penalty",
    type=int, help="Penalty", required=False)
parser.add_argument(
    "--eta",
    type=float, help="Eta", required=False)


args = parser.parse_args()
logging.info(args)

ContextProvider.create_context("file")
ContextProvider.get_context().set_features(DATASET_COLUMNS)
ContextProvider.get_context().set_experiment_id(args.id)
ContextProvider.get_context().set_working_dir(args.workdir)

DbProvider.create_context(
    context_type="file",
    d_0_path=args.dpath)

anotator = alf.anotator.AnotatorDoH(blacklist_path=args.blacklist)

if args.model == "single":
    model = alf.ml_model.SupervisedMLModel(VotingClassifier([
        ("rf1", RandomForestClassifier()),
        ("rf2", RandomForestClassifier()),
        ("rf3", RandomForestClassifier(criterion="entropy"))
    ], voting="soft"))
elif args.model == "committee":
    model = alf.ml_model.CommitteeMLModel(VotingClassifier([
        ("rf1", RandomForestClassifier()),
        ("rf2", RandomForestClassifier()),
        ("rf3", RandomForestClassifier(criterion="entropy"))
    ], voting="soft"))
else:
    raise ValueError("Unknown model name")

if args.query_strategy == "random":
    query_strategy = alf.query_strategy.RandomQueryStrategy(
        max_samples=args.query_nmax,
        anotator_obj=anotator,
        dry_run=True)
elif args.query_strategy == "entropy_ranked":
    query_strategy = alf.query_strategy.EntropyScoreRankedBatch(
        anotator_obj=anotator,
        max_samples=args.query_nmax,
        score_threshold=args.query_threshold,
        dry_run=True)
elif args.query_strategy == "entropy_unranked":
    query_strategy = alf.query_strategy.EntropyScoreRankedBatch(
        anotator_obj=anotator,
        max_samples=args.query_nmax,
        score_threshold=args.query_threshold,
        dry_run=True)
elif args.query_strategy == "uncertainty_ranked":
    query_strategy = alf.query_strategy.UncertanityRankedBatch(
        anotator_obj=anotator, max_samples=args.query_nmax,
        score_threshold=args.query_threshold, dry_run=True)
elif args.query_strategy == "uncertainty_unranked":
    query_strategy = alf.query_strategy.UncertanityUnrankedBatch(
        anotator_obj=anotator, max_samples=args.query_nmax,
        score_threshold=args.query_threshold, dry_run=True)
elif args.query_strategy == "density_unranked":
    query_strategy = alf.query_strategy.DensityUnrankedBatch(
        anotator_obj=anotator, max_samples=args.query_nmax,
        score_threshold=args.query_threshold, beta=args.beta,
        dry_run=True)
elif args.query_strategy == "density_ranked":
    query_strategy = alf.query_strategy.DensityRankedBatch(
        anotator_obj=anotator, max_samples=args.query_nmax,
        score_threshold=args.query_threshold, beta=args.beta,
        dry_run=True)
elif args.query_strategy == "kldiv":
    if not isinstance(model, alf.ml_model.CommitteeMLModel):
        raise ValueError("RAL query strategy requires a list of models")
    query_strategy = alf.query_strategy.KLDivergenceUnrankedBatch(
        anotator_obj=anotator, max_samples=args.query_nmax,
        score_threshold=args.query_threshold,
        dry_run=True)
elif args.query_strategy == "ral":
    if not isinstance(model, alf.ml_model.CommitteeMLModel):
        raise ValueError("RAL query strategy requires a list of models")
    query_strategy = alf.query_strategy.RAL(
        anotator_obj=anotator, dry_run=True, comittee_len=5,
        uncertainty_threshold=args.query_threshold,
        threshold_greedy=args.threshold_greedy, budget=args.budget,
        reward=args.reward, penalty=args.penalty, eta=args.eta)
else:
    raise ValueError("Unknown query strategy name")

if args.input == "folder":
    input_manager = alf.input_manager.TrapcapFolderInputManager(
        definition=args.input_def)
elif args.input == "socket":
    input_manager = alf.input_manager.TrapcapSocketInputManager(
        definition=args.input_def)
else:
    raise ValueError("Unknown input type")

if args.postprocessor == "undersample":
    postprocessor = alf.postprocess.PostprocessorUndersample()
else:
    postprocessor = alf.postprocess.PostprocessorIdentity()

engine = alf.engine.Engine(
    preprocessor=alf.preprocess.PreprocessorDoH(),
    postprocessor=postprocessor,
    ml_model_obj=model,
    query_strategy_obj=query_strategy,
    evaluator_obj=alf.evaluator.EvaluatorTestAnotatedAndAllPredicted(),
    input_manager_obj=input_manager
)
engine.run()
