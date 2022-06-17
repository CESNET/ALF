import random
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances

from . import anotator, context_manager, d_manager, ip_flow

DbProvider = d_manager.DbProvider
ContexProvider = context_manager.ContextProvider


class QueryStrategy(ABC):
    """QueryStrategy class provides interface for query strategies.
    """
    def __init__(
            self,
            anotator_obj: anotator.Anotator,
            dry_run=False,
            **options) -> None:
        """Initialize query strategy. Set oraculum/anotator and options.
        """
        self._oraculum = anotator_obj
        self._dry_run = dry_run

    @abstractmethod
    def select(
            self,
            class_proba: np.ndarray,
            flows: ip_flow.IPFlows,
            **options) -> ip_flow.IPFlows:
        """Select flows. Only public method. Need to be implemented by every
        strategy. Strategy gets class probabilities or list of class
        probabilities in case of committee based machine learning model, IP
        flows and classes list for mapping.

        Args:
            class_proba (np.ndarray): Class probabilities
            flows (ip_flow.IPFlows): IP flows
            classes (np.ndarray): Classes list

        Returns:
            ip_flow.IPFlows: Selected and anotated flows by given oracle during
            initialization.
        """

    def _anotate_selected(
            self,
            flows: ip_flow.IPFlows,
            selected: np.ndarray) -> ip_flow.IPFlows:
        """Anotate selected flows. If in dry run mode, anotate all.

        Args:
            flows (ip_flow.IPFlows): IP flows.
            selected (np.ndarray): Selected flows (mask or index array)

        Returns:
            ip_flow.IPFlows: Anotated IP flows.
        """
        if self._dry_run:
            return self._oraculum.anotate(flows), selected
        flows["class"] = None
        anotated = self._oraculum.anotate(
            flows.iloc[selected])
        flows.loc[selected, "class"] = anotated["class"]
        return flows, selected


class RandomQueryStrategy(QueryStrategy):
    """Implements query strategy based on random sample of flows.
    """
    def __init__(
        self,
            anotator_obj: anotator.Anotator, dry_run=False, **options) -> None:
        super().__init__(anotator_obj, dry_run, **options)
        self._max_samples = options.get("max_samples", 100)

    def select(
            self,
            class_proba: np.ndarray,
            flows: ip_flow.IPFlows,
            **options) -> ip_flow.IPFlows:
        """Random select of max_samples flows.

        Args:
            class_proba (np.ndarray): Do not need it here, just respect the
            interface.
            flows (ip_flow.IPFlows): List of flows predicted by ML model.
            max_samples (int): Number of random samples to be anotated.

        Returns:
            ip_flow.IPFlows: List of anotated flows.
        """
        if isinstance(self._max_samples, float):
            random_mask = np.random.choice(
                [True, False],
                size=len(flows),
                p=[self._max_samples, 1-self._max_samples])
            return self._anotate_selected(flows, random_mask)
        n = min(len(flows), self._max_samples)
        random_mask = np.zeros(len(flows), dtype=bool)
        rand_idxs = np.random.choice(len(random_mask), n, replace=False)
        random_mask[rand_idxs] = True
        return self._anotate_selected(flows, random_mask)


class ScoreAndBatchQueryStrategy(QueryStrategy):
    """Base class which batch selects flows based on scores. Can use score
    threshold which means, scores which are below threshold will not be
    queried and anotated. Most of strategies should work with the invariant
    that    low score is better for confidence so high score means very
    unconfident.
    """
    def __init__(
        self,
            anotator_obj: anotator.Anotator, dry_run=False, **options) -> None:
        super().__init__(anotator_obj, dry_run, **options)
        self._score_threshold = options.get("score_threshold", 0)
        self._max_samples = options.get("max_samples", 100)
        self._metric = options.get("metric", "euclidean")
        self._density_beta: float = options.get("beta", 1.0)

    def select(
            self,
            class_proba: np.ndarray,
            flows: ip_flow.IPFlows,
            **options) -> ip_flow.IPFlows:
        """Select flows based on scores and then batch them. Score and batch
        methods are both abstract. This select only prescribes the way how
        score and batch are combined.

        1. Score flows based on their features or most commonly based on
        class_proba which is P(c_i | x) list.

        2. Batch flows based on batch method (for example ranked batch).

        3. If score is bellow score, it is not queried and anotated.

        4. If score is above the score and batch decided to query it,
        it is anotated.
        """
        max_samples = min(len(class_proba), self._max_samples)
        metric = self._metric
        score_threshold = self._score_threshold
        scores = self._score(class_proba, flows=flows)
        above_threshold_flows = scores > score_threshold
        train_db = DbProvider.get_context().get_all()
        selection = self._batch(
            scores, flows, train_db, metric,
            max_samples)
        anotation_mask = above_threshold_flows & selection
        return self._anotate_selected(flows, anotation_mask)

    @abstractmethod
    def _score(self, class_proba: np.ndarray, **args) -> np.ndarray:
        """Give score the the flows.

        Args:
            class_proba (np.ndarray)

        Returns:
            np.ndarray
        """

    @abstractmethod
    def _batch(
            self, scores: np.ndarray, predicted: ip_flow.IPFlowsDataFrame,
            train: ip_flow.IPFlowsDataFrame, metric: str,
            max_samples: int) -> np.ndarray:
        """Batch flows based on score.

        Args:
            scores (np.ndarray)
            predicted (ip_flow.IPFlowsDataFrame)
            train (ip_flow.IPFlowsDataFrame)
            metric (str)
            max_samples (int)

        Returns:
            np.ndarray
        """


class RankedBatchQueryStrategy(ScoreAndBatchQueryStrategy):
    """Batch flows based on their score and non-similarity to training set.
    """

    def _batch(
            self, scores: np.ndarray, predicted: ip_flow.IPFlowsDataFrame,
            train: ip_flow.IPFlowsDataFrame, metric: str,
            max_samples: int) -> np.ndarray:
        available_indices = np.ones(len(predicted), np.bool8)
        ctx = ContexProvider.get_context()
        features = ctx.get_features()
        for _ in range(max_samples):
            a = len(predicted) / (len(predicted) + len(train))
            _, distances = pairwise_distances_argmin_min(
                predicted[features],
                train[features],
                metric=metric
            )
            sim_score = 1 / (1 + distances)
            final_score = \
                a * (1 - sim_score) + (1 - a) * scores
            # sorry for black magic
            # https://stackoverflow.com/questions/67930831/
            # numpy-original-index-from-masked-array
            avail, = np.where(available_indices)
            original_index = avail[final_score[avail].argmax()]
            available_indices[original_index] = False
        return ~available_indices


class UnrankedBatchQueryStrategy(ScoreAndBatchQueryStrategy):
    """Unranked batch query strategy. Just returns given flows based
    on their
    """

    def _batch(
            self, scores: np.ndarray, predicted: ip_flow.IPFlowsDataFrame,
            train: ip_flow.IPFlowsDataFrame, metric: str,
            max_samples: int) -> np.ndarray:
        if max_samples >= len(scores):
            return np.ones(len(scores), dtype=bool)
        top_n = np.argpartition(scores, -max_samples)[-max_samples:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_n] = 1
        return mask


class InformationDensityUncertScore(ScoreAndBatchQueryStrategy):
    """Score based on uncertainty and information density scoring.
    """
    def _score(self, class_proba: np.ndarray, **args) -> np.ndarray:
        ctx = ContexProvider.get_context()
        features = ctx.get_features()
        uncertainty = 1 - np.max(class_proba, axis=1)
        unlabeled = args.get("flows", None)[features]
        distances = pairwise_distances(
            unlabeled, unlabeled, metric=self._metric)
        similarities = 1 / (1 + distances)
        return uncertainty * similarities.mean(axis=1)**self._density_beta


class UncertaintyScore(ScoreAndBatchQueryStrategy):
    """Score based on uncertainty.
    """
    def _score(self, class_proba: np.ndarray, **args) -> np.ndarray:
        score = 1 - np.max(class_proba, axis=1)
        return score


class SmallestMarginScore(ScoreAndBatchQueryStrategy):
    """Score based on smallest margin.
    """
    def _score(self, class_proba: np.ndarray, **args) -> np.ndarray:
        sorted_proba = np.partition(class_proba, 1, axis=1)
        score = sorted_proba[:, -1] - sorted_proba[:, 0]
        return score


class EntropyScore(ScoreAndBatchQueryStrategy):
    """Score based on entropy or distribution.
    """
    def _score(self, class_proba: np.ndarray, **args) -> np.ndarray:
        score = entropy(class_proba, axis=1)
        return score


class KLDivergenceComitteeScore(ScoreAndBatchQueryStrategy):
    """Score based on KL divergence between two distributions.
    """
    def _score(self, class_proba: np.ndarray, **args) -> np.ndarray:
        P_C = class_proba.mean(axis=1)
        KL_X = np.zeros(class_proba.shape[0])
        for x_i in range(class_proba.shape[0]):
            for C_i in range(class_proba.shape[1]):
                KL_X[x_i] = entropy(
                    pk=class_proba[x_i, C_i, :],
                    qk=P_C[x_i]
                )
        return KL_X/class_proba.shape[1]


class EntropyScoreRankedBatch(EntropyScore, RankedBatchQueryStrategy):
    """Query strategy based on entropy scoring and ranked batch.
    """


class UncertanityRankedBatch(UncertaintyScore, RankedBatchQueryStrategy):
    """Query strategy based on uncertainty scoring and ranked batch.
    """


class KLDivergenceRankedBatch(
    KLDivergenceComitteeScore,
        RankedBatchQueryStrategy):
    """Query strategy based on KL divergance scoring and ranked batch.
    """


class KLDivergenceUnrankedBatch(
    KLDivergenceComitteeScore,
        UnrankedBatchQueryStrategy):
    """Query strategy based on KL divergance scoring and unranked batch.
    """


class MarginRankedBatch(SmallestMarginScore, RankedBatchQueryStrategy):
    """Query strategy based on smallest margin scoring and ranked batch.
    """


class UncertanityUnrankedBatch(UncertaintyScore, UnrankedBatchQueryStrategy):
    """Query strategy based on uncertainty scoring and unranked batch.
    """


class EntropyUnrankedBatch(EntropyScore, UnrankedBatchQueryStrategy):
    """Query strategy based on entropy scoring and unranked batch.
    """


class DensityUnrankedBatch(
    InformationDensityUncertScore,
        UnrankedBatchQueryStrategy):
    """Strategy based on information density with uncertainty score
    with unranked batch
    """


class DensityRankedBatch(
    InformationDensityUncertScore,
        RankedBatchQueryStrategy):
    """Strategy based on information density with uncertainty score
    with ranked batch
    """


class RAL(QueryStrategy):
    """Reinforcement Active Learning strategy as described at
    https://hal.archives-ouvertes.fr/hal-02265426/document
    """
    def __init__(
        self,
            anotator_obj: anotator.Anotator,
            dry_run=False,
            uncertainty_threshold=0.9,
            threshold_greedy=0.025,
            comittee_len=-1,
            budget=0.05,
            reward=1, penalty=1, eta=0.01) -> None:
        if comittee_len < 1:
            raise ValueError("comittee_len must be specified and > 0")
        super().__init__(anotator_obj=anotator_obj, dry_run=dry_run)
        self._uncertainty_threshold = uncertainty_threshold
        self._a = [1.0 / comittee_len for _ in range(comittee_len)]
        self._budget = budget
        self._n_acquired_samples = 0
        self._n_seen = 0
        self._threshold_greedy = threshold_greedy
        self._reward = reward
        self._penalty = penalty
        self._eta = eta
        self._n_committee_too_low_acquires = 0
        self._n_committee_acquires = 0
        self._n_greedy_acquires = 0
        self._n_both_acquires = 0
        self._n_seen = 0
        return

    def _update_alpha(self, reward, decisions, committee_decision):
        for idx, decision in enumerate(decisions):
            if int(decision) == int(committee_decision):
                self._a[idx] = \
                    self._a[idx] * np.exp(reward * self._eta)
        s = sum(self._a)
        for idx, coef in enumerate(self._a):
            self._a[idx] = float(coef) / s

    def _update_uncertainty_threshold(self, reward):
        f_reward = \
            self._eta * (1.0 - np.power(2.0, float(reward) / self._penalty))
        self._uncertainty_threshold = \
            max(min(self._uncertainty_threshold * (1.0 + f_reward), 1.0), 0.6)

    def _get_reward(self, y_true, y_predicted):
        if y_true != y_predicted:
            return self._reward
        else:
            return self._penalty

    def _ask_certainty(self, P_C: np.ndarray):
        decisions = []
        for P_Ci in P_C:
            decisions.append(np.max(P_Ci) < self._uncertainty_threshold)
        comittee_decision = \
            bool(round(sum(
                [self._a[idx] * el for idx, el in enumerate(decisions)]
            )))
        return comittee_decision, decisions

    def _has_enough_budget(self):
        if self._budget is None:
            return True
        return float(self._n_acquired_samples) / self._n_seen < self._budget

    def _labeling_decision(self, committee_decision):
        self._n_seen += 1
        if not self._has_enough_budget():
            if committee_decision:
                self._n_committee_too_low_acquires += 1
            return False
        if committee_decision:
            self._n_committee_acquires += 1
        if random.random() < self._threshold_greedy:  # nosec
            self._n_greedy_acquires += 1
            if committee_decision:
                self._n_both_acquires += 1
            return True
        else:
            return committee_decision

    def _get_label(self, flow, pred_y, committee_decision, decisions):
        true_y = self._oraculum.anotate(flow)['class'].iloc[0]
        if committee_decision:
            reward = self._get_reward(true_y, pred_y)
            self._update_alpha(reward, decisions, committee_decision)
            self._update_uncertainty_threshold(reward)
        self._n_acquired_samples += 1
        return true_y

    def select(
            self,
            class_proba: np.ndarray,
            flows: ip_flow.IPFlowsDataFrame,
            **options) -> ip_flow.IPFlowsDataFrame:
        classes = options.get("classes")
        return_mask = np.zeros(len(class_proba), np.bool8)
        pred_ys = class_proba.mean(axis=1)
        true_classes = []
        for i, x_i in enumerate(class_proba):
            pred_y = classes[np.argmax(pred_ys[i])]
            flow = flows.iloc[[i]]
            committeeDecision, learnerDecisions = \
                self._ask_certainty(x_i)
            final_decision = self._labeling_decision(committeeDecision)
            if final_decision:
                true_class = self._get_label(
                    flow, pred_y, committeeDecision, learnerDecisions)
                true_classes.append(true_class)
                return_mask[i] = True
        ctx = ContexProvider.get_context()
        ctx.append_metrics({
            "ral": {
                "uncertainty_threshold": self._uncertainty_threshold,
                "alpha": self._a,
                "n_acquired_samples": self._n_acquired_samples,
                "n_seen": self._n_seen,
                "n_committee_too_low_acquires":
                    self._n_committee_too_low_acquires,
                "n_committee_acquires": self._n_committee_acquires,
                "n_greedy_acquires": self._n_greedy_acquires,
                "n_both_acquires": self._n_both_acquires
            }
        })
        # if dry run mode, let anotator anotate all flows and return it
        if self._dry_run:
            return self._anotate_selected(flows, return_mask)
        # if not dry run mode, return only selected flows
        flows["class"] = None
        flows.loc[return_mask, "class"] = true_classes
        return flows, return_mask
