import math

import networkx as nx
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.bivariate_scores import get_bivariate_score
from pgmpy.utils import get_dataset_type

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


class NormFlow(BaseEstimator):
    r"""
    Nonlinear-ICA disturbance estimator for the Post-Nonlinear model.

    For observations in ``[cause, effect]`` order, the estimator jointly learns
    an inner mechanism :math:`l_1` and a smooth, strictly increasing transform
    :math:`l_2` and returns

    .. math:: e = l_2(effect) - l_1(cause).

    The objective is the negative nonlinear-ICA log-likelihood from the PNL
    identification method:

    .. math:: -\log p_e(e) - \log l_2'(effect).

    A small Gaussian mixture models the otherwise unspecified one-dimensional
    disturbance density. The post-transform consists of a positive linear term
    plus positive weighted ``tanh`` units, so its derivative is positive for all
    inputs and the transform is globally invertible.

    Parameters
    ----------
    hidden_dim : int, default=12
        Number of hidden units in the inner mechanism and post-transform.

    n_components : int, default=5
        Number of components in the disturbance-density Gaussian mixture.

    max_iter : int, default=1000
        Number of full-batch optimization steps.

    learning_rate : float, default=1e-3
        Adam learning rate.

    seed : int, optional
        PyTorch random seed.
    """

    def __init__(self, hidden_dim=12, n_components=5, max_iter=1000, learning_rate=1e-3, seed=None):
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.seed = seed

    def _post_transform(self, effect):
        """Evaluate the monotone post-transform and its positive derivative."""
        input_weight = torch.nn.functional.softplus(self.post_input_weight_)
        output_weight = torch.nn.functional.softplus(self.post_output_weight_)
        slope = torch.nn.functional.softplus(self.post_slope_) + 1e-4

        hidden = torch.tanh(effect * input_weight + self.post_bias_)
        transformed = self.post_intercept_ + slope * effect + hidden @ output_weight.T
        derivative = slope + ((1.0 - hidden**2) * input_weight * output_weight).sum(dim=1, keepdim=True)
        return transformed, derivative

    def _negative_log_density(self, disturbance):
        """Evaluate the learned Gaussian-mixture negative log density."""
        scale = torch.nn.functional.softplus(self.noise_scale_) + 1e-4
        standardized = (disturbance - self.noise_mean_) / scale

        component_log_prob = -0.5 * standardized**2 - torch.log(scale) - 0.5 * math.log(2.0 * math.pi)
        log_weight = torch.log_softmax(self.noise_logits_, dim=0)
        return -torch.logsumexp(component_log_prob + log_weight, dim=1).mean()

    def fit(self, X, y=None):
        """
        Fit the PNL functions to data in ``[cause, effect]`` column order.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Candidate cause and effect observations.

        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : NormFlow
            The fitted estimator.
        """
        # Step 1: Validate the dependency, input shape, and hyperparameters.
        _check_soft_dependencies("torch", obj=self)

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"NormFlow expects a 2-column array, got shape {X.shape}.")
        if self.hidden_dim < 1 or self.n_components < 1 or self.max_iter < 1:
            raise ValueError("hidden_dim, n_components, and max_iter must all be positive integers.")

        # Step 2: Record scaling statistics and reject constant columns.
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        if np.any(self.scale_ == 0):
            raise ValueError("NormFlow requires non-constant cause and effect columns.")

        # Step 3: Standardize the data and convert it to tensors.
        if self.seed is not None:
            torch.manual_seed(self.seed)

        standardized = (X - self.mean_) / self.scale_
        cause_t = torch.tensor(standardized[:, 0:1], dtype=torch.float32)
        effect_t = torch.tensor(standardized[:, 1:2], dtype=torch.float32)

        # Step 4: Initialize the mechanism, transform, and noise parameters.
        self.inner_model_ = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.post_input_weight_ = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.post_output_weight_ = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.post_bias_ = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.post_slope_ = nn.Parameter(torch.zeros(1, 1))
        self.post_intercept_ = nn.Parameter(torch.zeros(1, 1))
        self.noise_logits_ = nn.Parameter(torch.zeros(self.n_components))
        self.noise_mean_ = nn.Parameter(torch.linspace(-1.0, 1.0, self.n_components))
        self.noise_scale_ = nn.Parameter(torch.zeros(self.n_components))

        # Step 5: Collect all trainable parameters and create the optimizer.
        parameters = list(self.inner_model_.parameters()) + [
            self.post_input_weight_,
            self.post_output_weight_,
            self.post_bias_,
            self.post_slope_,
            self.post_intercept_,
            self.noise_logits_,
            self.noise_mean_,
            self.noise_scale_,
        ]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

        # Step 6: Optimize the PNL likelihood objective.
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            transformed_effect, derivative = self._post_transform(effect_t)
            disturbance = transformed_effect - self.inner_model_(cause_t)
            loss = self._negative_log_density(disturbance) - torch.log(derivative).mean()
            if not torch.isfinite(loss):
                raise ValueError("PNL optimization produced a non-finite loss.")
            loss.backward()
            optimizer.step()

        # Step 7: Store the final optimization loss.
        self.loss_ = float(loss.detach())
        return self

    def predict(self, X):
        """
        Estimate the PNL disturbance for each observation.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Candidate cause and effect observations.

        Returns
        -------
        disturbance : np.ndarray of shape (n_samples,)
            Estimated disturbance values.
        """
        check_is_fitted(self, attributes=["inner_model_", "mean_", "scale_"])
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"NormFlow expects a 2-column array, got shape {X.shape}.")

        standardized = (X - self.mean_) / self.scale_
        cause_t = torch.tensor(standardized[:, 0:1], dtype=torch.float32)
        effect_t = torch.tensor(standardized[:, 1:2], dtype=torch.float32)
        with torch.no_grad():
            transformed_effect, _ = self._post_transform(effect_t)
            disturbance = transformed_effect - self.inner_model_(cause_t)
        return disturbance.cpu().numpy().ravel()


class PNL(BaseCausalDiscovery):
    r"""
    Bivariate causal discovery using the Post-Nonlinear model.

    PNL assumes ``effect = f2(f1(cause) + disturbance)``, with an invertible
    post-nonlinearity ``f2`` and a disturbance independent of the cause. It fits
    the disturbance model in both directions and chooses the direction with the
    smaller dependence score.

    Parameters
    ----------
    estimator : sklearn-compatible estimator, default=None
        Cloneable estimator accepting two columns in ``[cause, effect]`` order
        and predicting the recovered disturbance. If ``None``, :class:`NormFlow`
        is used and the optional PyTorch dependency is required.

    score : str, BaseBivariateScore, or callable, default="independence"
        Dependence score applied to the candidate cause and recovered
        disturbance. A smaller value indicates the preferred direction.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        Learned graph with one directed edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix of ``causal_graph_``.

    forward_score_, backward_score_ : float
        Dependence scores for the two input-column directions.

    forward_estimator_, backward_estimator_ : estimator
        Fitted disturbance estimators for both directions.

    References
    ----------
    - Zhang, K., and Hyvarinen, A. On the identifiability of the post-nonlinear
      causal model. UAI, 2009.
    """

    def __init__(self, estimator=None, score="independence"):
        self.estimator = estimator
        self.score = score

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame):
        """Fit the PNL model in both possible directions."""
        # Step 1: Validate that the input contains two continuous variables.
        if X.shape[1] != 2:
            raise ValueError(f"PNL requires exactly two variables, got {X.shape[1]}.")
        if get_dataset_type(X) != "continuous":
            raise ValueError("PNL requires continuous (numeric) variables; got non-continuous data.")

        # Step 2: Ensure that neither variable is constant.
        x, y = self.feature_names_in_
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; PNL requires non-constant variables.")

        # Step 3: Fit and score the forward and backward causal directions.
        score = get_bivariate_score(self.score, algorithm="pnl")
        self.forward_score_, self.forward_estimator_ = self._direction_score(X[[x, y]], score)
        self.backward_score_, self.backward_estimator_ = self._direction_score(X[[y, x]], score)

        # Step 4: Select the lower-scoring direction and construct its graph.
        edge = (x, y) if self.forward_score_ <= self.backward_score_ else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")
        return self

    def _direction_score(self, pair: pd.DataFrame, score):
        """Fit and score one candidate direction."""
        estimator = NormFlow() if self.estimator is None else clone(self.estimator)
        estimator.fit(pair)
        disturbance = np.asarray(estimator.predict(pair)).ravel()
        if not np.isfinite(disturbance).all():
            raise ValueError("PNL estimator produced non-finite disturbance predictions.")

        direction_score = float(score(pair.iloc[:, 0].to_numpy(), disturbance))
        if not np.isfinite(direction_score):
            raise ValueError("PNL dependence score is non-finite.")
        return direction_score, estimator
