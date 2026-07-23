"""
OpenMax EVT fitting and inference utilities.

The Weibull fitting implementation is derived from the OpenMax reference code,
while this module exposes a Pointcept-specific datadict API.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from . import pairwise_distances
from .weibull import weibull


OPENMAX_MODEL_VERSION = 1


def fit_high(distances, distance_multiplier, tailsize, translate_amount=1):
    """Fit the high tail exactly as in the VAST OpenMax implementation."""
    if tailsize <= 0:
        raise ValueError(f"tailsize must be positive, got {tailsize}")
    if tailsize <= 1:
        tailsize = min(tailsize * distances.shape[1], distances.shape[1])
    tailsize = int(min(tailsize, distances.shape[1]))
    if tailsize < 1:
        raise ValueError("tailsize selects no activation distances")
    mr = weibull(translateAmount=translate_amount)
    mr.FitHigh(distances.double() * distance_multiplier, tailsize, isSorted=False)
    mr.tocpu()
    return mr


def collect_correct_activations(
    activations: torch.Tensor,
    segment: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """
    Group correctly classified activation vectors by ground-truth class.

    Invalid/ignored labels are excluded.  Returned tensors are detached and
    moved to CPU so callers can accumulate a full training set without holding
    its autograd graph or consuming GPU memory.
    """
    if activations.ndim != 2:
        raise ValueError(
            "activations must have shape [num_points, num_classes], "
            f"got {tuple(activations.shape)}"
        )
    segment = segment.reshape(-1)
    if segment.shape[0] != activations.shape[0]:
        raise ValueError(
            "segment and activations must contain the same number of points, "
            f"got {segment.shape[0]} and {activations.shape[0]}"
        )
    if num_classes is None:
        num_classes = activations.shape[1]
    if num_classes != activations.shape[1]:
        raise ValueError(
            f"num_classes={num_classes} does not match activation dimension "
            f"{activations.shape[1]}"
        )

    prediction = activations.argmax(dim=1)
    valid = (segment >= 0) & (segment < num_classes)
    correct = valid & prediction.eq(segment)
    result = {}
    for class_id in segment[correct].unique(sorted=True).tolist():
        class_mask = correct & segment.eq(class_id)
        result[int(class_id)] = activations[class_mask].detach().cpu()
    return result


def _distance_function(name):
    try:
        return pairwise_distances.DISTANCE_FUNCTIONS[name]
    except KeyError as error:
        choices = ", ".join(sorted(pairwise_distances.DISTANCE_FUNCTIONS))
        raise ValueError(
            f"Unsupported OpenMax distance metric '{name}'. Choose from: {choices}"
        ) from error


def fit_openmax_class(
    activations: torch.Tensor,
    class_id: int,
    tailsize: float = 1.0,
    distance_multiplier: float = 1.0,
    translate_amount: float = 1.0,
    distance_metric: str = "cosine",
    device: Optional[torch.device] = None,
):
    """Fit the MAV and Weibull model for one class."""
    if activations.ndim != 2:
        raise ValueError(
            f"Class {class_id} activations must be 2-D, "
            f"got shape {tuple(activations.shape)}"
        )
    if activations.shape[0] == 0:
        raise ValueError(
            "Cannot fit OpenMax without correctly classified activations for "
            f"class {class_id}"
        )
    if distance_multiplier <= 0:
        raise ValueError(
            "distance_multiplier must be positive, "
            f"got {distance_multiplier}"
        )

    device = torch.device("cpu") if device is None else torch.device(device)
    activations = activations.detach().to(
        device=device, dtype=torch.double
    )
    mav = torch.mean(activations, dim=0)
    distances = _distance_function(distance_metric)(
        activations, mav[None, :]
    )
    weibull_model = fit_high(
        distances.T,
        distance_multiplier=distance_multiplier,
        tailsize=tailsize,
        translate_amount=translate_amount,
    )
    return {
        "class_id": int(class_id),
        "activation_dim": int(activations.shape[1]),
        "activation_count": int(activations.shape[0]),
        "model": {
            "mav": mav.detach().cpu()[None, :],
            "weibull": weibull_model,
        },
    }


def assemble_openmax_models(
    fitted_classes: Mapping[int, dict],
    tailsize: float = 1.0,
    distance_multiplier: float = 1.0,
    translate_amount: float = 1.0,
    distance_metric: str = "cosine",
):
    """Assemble independently fitted classes into an OpenMax model datadict."""
    class_ids = sorted(int(class_id) for class_id in fitted_classes)
    if not class_ids:
        raise ValueError("No fitted OpenMax classes were provided")

    activation_dims = {
        int(fitted_classes[class_id]["activation_dim"])
        for class_id in class_ids
    }
    if len(activation_dims) != 1:
        raise ValueError(
            "All classes must use the same activation dimension, "
            f"got {sorted(activation_dims)}"
        )
    for class_id in class_ids:
        fitted_class_id = int(fitted_classes[class_id]["class_id"])
        if fitted_class_id != class_id:
            raise ValueError(
                f"Fitted class {fitted_class_id} was stored as class "
                f"{class_id}"
            )

    return {
        "version": OPENMAX_MODEL_VERSION,
        "metadata": {
            "class_ids": class_ids,
            "activation_dim": activation_dims.pop(),
            "distance_metric": distance_metric,
            "tailsize": float(tailsize),
            "distance_multiplier": float(distance_multiplier),
            "translate_amount": float(translate_amount),
            "activation_counts": {
                class_id: int(
                    fitted_classes[class_id]["activation_count"]
                )
                for class_id in class_ids
            },
        },
        "models": {
            class_id: fitted_classes[class_id]["model"]
            for class_id in class_ids
        },
    }


def fit_openmax_models(
    activations_by_class: Mapping[int, torch.Tensor],
    tailsize: float = 1.0,
    distance_multiplier: float = 1.0,
    translate_amount: float = 1.0,
    distance_metric: str = "cosine",
    class_ids: Optional[Iterable[int]] = None,
    device: Optional[torch.device] = None,
):
    """
    Fit one mean activation vector and Weibull EVT model per class.

    Returns a runtime datadict accepted by :func:`openmax_inference` and
    :func:`save_openmax_models`.
    """
    if class_ids is None:
        class_ids = sorted(int(class_id) for class_id in activations_by_class)
    else:
        class_ids = sorted(int(class_id) for class_id in class_ids)
    if not class_ids:
        raise ValueError("No classes were provided for OpenMax fitting")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError(f"class_ids contains duplicates: {class_ids}")

    missing = [
        class_id
        for class_id in class_ids
        if class_id not in activations_by_class
        or activations_by_class[class_id].numel() == 0
    ]
    if missing:
        raise ValueError(
            "Cannot fit OpenMax without correctly classified activations for "
            f"every class. Missing classes: {missing}"
        )

    fitted_classes = {
        class_id: fit_openmax_class(
            activations=activations_by_class[class_id],
            class_id=class_id,
            tailsize=tailsize,
            distance_multiplier=distance_multiplier,
            translate_amount=translate_amount,
            distance_metric=distance_metric,
            device=device,
        )
        for class_id in class_ids
    }
    return assemble_openmax_models(
        fitted_classes=fitted_classes,
        tailsize=tailsize,
        distance_multiplier=distance_multiplier,
        translate_amount=translate_amount,
        distance_metric=distance_metric,
    )


def _serializable_openmax_models(fitted_models):
    models = {}
    for class_id, class_model in fitted_models["models"].items():
        models[int(class_id)] = {
            "mav": class_model["mav"].detach().cpu(),
            "weibull": {
                name: value.detach().cpu() if torch.is_tensor(value) else value
                for name, value in class_model[
                    "weibull"
                ].return_all_parameters().items()
            },
        }
    return {
        "version": int(fitted_models["version"]),
        "metadata": dict(fitted_models["metadata"]),
        "models": models,
    }


def save_openmax_models(fitted_models, path):
    """Save a fitted OpenMax datadict without pickling Weibull objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_serializable_openmax_models(fitted_models), path)


def load_openmax_models(path):
    """Load tensor-only OpenMax state and reconstruct its Weibull models."""
    path = Path(path)
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 2.0 does not provide the weights_only argument.
        saved = torch.load(path, map_location="cpu")
    if saved.get("version") != OPENMAX_MODEL_VERSION:
        raise ValueError(
            f"Unsupported OpenMax model version {saved.get('version')}; "
            f"expected {OPENMAX_MODEL_VERSION}"
        )

    models = {}
    for class_id, class_model in saved["models"].items():
        models[int(class_id)] = {
            "mav": class_model["mav"],
            "weibull": weibull(class_model["weibull"]),
        }
    fitted_models = {
        "version": saved["version"],
        "metadata": saved["metadata"],
        "models": models,
    }
    _validate_fitted_models(fitted_models)
    return fitted_models


def _validate_fitted_models(fitted_models, activation_dim=None):
    metadata = fitted_models["metadata"]
    class_ids = [int(class_id) for class_id in metadata["class_ids"]]
    missing = [
        class_id
        for class_id in class_ids
        if class_id not in fitted_models["models"]
    ]
    if missing:
        raise ValueError(f"OpenMax model is missing fitted classes: {missing}")
    if activation_dim is not None and metadata["activation_dim"] != activation_dim:
        raise ValueError(
            "OpenMax activation dimension does not match the segmentor output: "
            f"{metadata['activation_dim']} != {activation_dim}"
        )
    if activation_dim is not None and class_ids != list(range(activation_dim)):
        raise ValueError(
            "OpenMax requires one fitted model for every activation column in "
            f"zero-based order. Got class_ids={class_ids}"
        )
    return class_ids


def openmax_inference(activations, fitted_models):
    """
    Compute per-class EVT knownness for an activation matrix.

    The returned ``evt_knownness`` columns follow ``class_ids``.
    """
    if activations.ndim != 2:
        raise ValueError(
            "activations must have shape [num_points, num_classes], "
            f"got {tuple(activations.shape)}"
        )
    class_ids = _validate_fitted_models(
        fitted_models, activation_dim=activations.shape[1]
    )
    distance_fn = _distance_function(
        fitted_models["metadata"]["distance_metric"]
    )
    double_activations = activations.double()
    knownness = []
    for class_id in class_ids:
        class_model = fitted_models["models"][class_id]
        mav = class_model["mav"].to(
            device=activations.device, dtype=double_activations.dtype
        )
        distances = distance_fn(double_activations, mav)
        knownness.append(
            class_model["weibull"].wscore(distances, isReversed=True)
        )
    evt_knownness = torch.cat(knownness, dim=1).to(dtype=activations.dtype)
    return {
        "class_ids": class_ids,
        "evt_knownness": evt_knownness,
    }


def openmax_alpha(
    evt_probs,
    activations,
    alpha=1,
    run_paper_version=True,
    *args,
    **kwargs,
):
    """
    Apply the OpenMax alpha rank weighting from the VAST implementation.

    The unknown class is column zero in ``logits`` and ``probabilities``.
    ``prediction`` uses the segmentor's zero-based class ids and ``-1`` for
    unknown.
    """
    if evt_probs.shape != activations.shape:
        raise ValueError(
            "evt_probs and activations must have the same shape, got "
            f"{tuple(evt_probs.shape)} and {tuple(activations.shape)}"
        )
    if activations.ndim != 2:
        raise ValueError("evt_probs and activations must both be 2-D")
    if not isinstance(alpha, int) or alpha < 1:
        raise ValueError(f"alpha must be a positive integer, got {alpha}")
    alpha = min(alpha, activations.shape[1])

    per_class_unknownness_prob = 1 - evt_probs
    sorted_activations, indices = torch.sort(
        activations, descending=True, dim=1
    )
    weights = torch.ones_like(activations)

    rank = torch.arange(
        1,
        alpha + 1,
        device=activations.device,
        dtype=activations.dtype,
    )
    if run_paper_version:
        rank = (alpha - rank) / alpha
    else:
        rank = ((alpha + 1) - rank) / alpha
    weights[:, :alpha] = 1 - rank * torch.gather(
        per_class_unknownness_prob, 1, indices[:, :alpha]
    )

    sorted_revised_activations = sorted_activations * weights
    unknown_activation = torch.sum(
        sorted_activations * (1 - weights), dim=1
    )
    revised_activations = torch.scatter(
        torch.ones_like(sorted_revised_activations),
        1,
        indices,
        sorted_revised_activations,
    )
    logits = torch.cat(
        [unknown_activation[:, None], revised_activations], dim=1
    )
    probabilities = F.softmax(logits, dim=1)
    score, openmax_class = torch.max(probabilities, dim=1)
    prediction = openmax_class - 1
    legacy_score = score.clone()
    legacy_score[openmax_class == 0] = -1.0

    return {
        "logits": logits,
        "probabilities": probabilities,
        "unknown_probability": probabilities[:, 0],
        "known_probabilities": probabilities[:, 1:],
        "revised_activations": revised_activations,
        "prediction": prediction,
        "score": score,
        "legacy_score": legacy_score,
    }


def apply_openmax(
    activations,
    fitted_models,
    alpha=1,
    run_paper_version=True,
):
    """Run EVT inference followed by OpenMax alpha reweighting."""
    evt_result = openmax_inference(activations, fitted_models)
    alpha_result = openmax_alpha(
        evt_probs=evt_result["evt_knownness"],
        activations=activations,
        alpha=alpha,
        run_paper_version=run_paper_version,
    )
    return {
        "class_ids": evt_result["class_ids"],
        "evt_knownness": evt_result["evt_knownness"],
        **alpha_result,
    }
