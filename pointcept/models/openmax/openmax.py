"""
Author: Akshay Raj Dhamija

@inproceedings{bendale2016towards,
  title={Towards open set deep networks},
  author={Bendale, Abhijit and Boult, Terrance E},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1563--1572},
  year={2016}
}

While the reimplementation in this file only performs the EVT-recognition part of the original paper, for the actual
openmax algorithm use the functionality from this file in conjunction with openmax_alpha function at the end
"""
import itertools
from typing import Iterator, Tuple, List, Dict

import torch
import torch.nn.functional as F

from .pairwise_distances import pairwisedistances
from .weibull import weibull


def OpenMax_Params(parser):
    OpenMax_params = parser.add_argument_group("OpenMax params")
    OpenMax_params.add_argument(
        "--tailsize",
        nargs="+",
        type=float,
        default=[1.0],
        help="tail size to use default: %(default)s",
    )
    OpenMax_params.add_argument(
        "--distance_multiplier",
        nargs="+",
        type=float,
        default=[1.0],
        help="distance multiplier to use default: %(default)s",
    )
    OpenMax_params.add_argument(
        "--translateAmount",
        nargs="+",
        type=float,
        default=1.0,
        help="translateAmount to use default: %(default)s",
    )
    OpenMax_params.add_argument(
        "--distance_metric",
        default="cosine",
        type=str,
        choices=["cosine", "euclidean"],
        help="distance metric to use default: %(default)s",
    )
    return parser, dict(
        group_parser=OpenMax_params,
        param_names=("tailsize", "distance_multiplier"),
        param_id_string="TS_{}_DM_{:.2f}",
    )


def fit_high(distances, distance_multiplier, tailsize, translateAmount=1):
    if tailsize <= 1:
        tailsize = min(tailsize * distances.shape[1], distances.shape[1])
    tailsize = int(min(tailsize, distances.shape[1]))
    mr = weibull.weibull(translateAmount=translateAmount)
    mr.FitHigh(distances.double() * distance_multiplier, tailsize, isSorted=False)
    mr.tocpu()
    return mr


def OpenMax_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None,
) -> Iterator[Tuple[str, Tuple[str, dict]]]:
    """
    :param pos_classes_to_process: List of class names to be processed by this function in the current process.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of the keys for this dictionary
    :param args: A named tuple or an argument parser object containing the arguments mentioned in the EVM_Params function.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: Not used during training, input ignored.
    :return: Iterator(Tuple(parameter combination identifier, Tuple(class name, its evm model)))
    """
    if "translateAmount" not in args.__dict__:
        args.translateAmount = 1
    device = "cpu" if gpu == -1 else f"cuda:{gpu}"
    for pos_cls_name in pos_classes_to_process:
        features = features_all_classes[pos_cls_name].clone().to(device)
        MAV = torch.mean(features, dim=0).to(device)
        distances = pairwisedistances.__dict__[args.distance_metric](
            features, MAV[None, :]
        )
        for tailsize, distance_multiplier in itertools.product(
            args.tailsize, args.distance_multiplier
        ):
            weibull_model = fit_high(
                distances.T, distance_multiplier, tailsize, args.translateAmount
            )
            yield (
                f"TS_{tailsize}_DM_{distance_multiplier:.2f}",
                (pos_cls_name, dict(MAV=MAV.cpu()[None, :], weibulls=weibull_model)),
            )


def OpenMax_Inference(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models: Dict = None,
) -> Iterator[Tuple[str, Tuple[str, torch.Tensor]]]:
    """
    :param pos_classes_to_process: List of batches to be processed by this function in the current process.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of
                                the keys for this dictionary
    :param args: Can be a named tuple or an argument parser object containing the arguments mentioned in the EVM_Params
                function above. Only the distance_metric argument is actually used during inferencing.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: The collated model created for a single hyper parameter combination.
    :return: Iterator(Tuple(str, Tuple(batch_identifier, torch.Tensor)))
    """
    device = "cpu" if gpu == -1 else f"cuda:{gpu}"
    for batch_to_process in pos_classes_to_process:
        features = features_all_classes[batch_to_process].to(device)
        probs = []
        for class_name in sorted(models.keys()):
            MAV = models[class_name]["MAV"].double().to(device)
            distances = pairwisedistances.__dict__[args.distance_metric](features, MAV)
            probs.append(models[class_name]["weibulls"].wscore(distances.cpu(), isReversed=True))
        probs = torch.cat(probs, dim=1)
        yield ("probs", (batch_to_process, probs))


def openmax_alpha(
    evt_probs, activations, alpha=1, run_paper_version=True, *args, **kwargs
):
    """
    Algorithm 2 OpenMax probability estimation with rejection of
    unknown or uncertain inputs.
    Require: Activation vector for v(x) = v1(x), . . . , vN (x)
    Require: means µj and libMR models ρj = (τi, λi, κi)
    Require: α, the numer of “top” classes to revise
    1: Let s(i) = argsort(vj (x)); Let ωj = 1
    2: for i = 1, . . . , α do
    3:     ωs(i)(x) = 1 − ((α−i)/α)*e^−((||x−τs(i)||/λs(i))^κs(i))
    4: end for
    5: Revise activation vector vˆ(x) = v(x) ◦ ω(x)
    6: Define vˆ0(x) = sum_i vi(x)(1 − ωi(x)).
    7:     Pˆ(y = j|x) = eˆvj(x)/sum_{i=0}_N eˆvi(x)
    8: Let y∗ = argmaxj P(y = j|x)
    9: Reject input if y∗ == 0 or P(y = y∗|x) < ǫ
    """
    # convert weibull CDF probabilities from knownness per class to unknownness per class
    per_class_unknownness_prob = 1 - evt_probs

    # Line 1
    sorted_activations, indices = torch.sort(activations, descending=True, dim=1)
    weights = torch.ones(activations.shape[0], activations.shape[1])

    # Line 2-4
    weights[:, :alpha] = torch.arange(1, alpha + 1, step=1)
    if run_paper_version:
        weights[:, :alpha] = (alpha - weights[:, :alpha]) / alpha
    else:
        # The version in the code is slightly different from the algorithm mentioned in the paper
        weights[:, :alpha] = ((alpha + 1) - weights[:, :alpha]) / alpha
    weights[:, :alpha] = 1 - weights[:, :alpha] * torch.gather(
        per_class_unknownness_prob, 1, indices[:, :alpha]
    )

    # Line 5
    revisted_activations = sorted_activations * weights
    # Line 6
    unknowness_class_prob = torch.sum(sorted_activations * (1 - weights), dim=1)
    revisted_activations = torch.scatter(
        torch.ones(revisted_activations.shape), 1, indices, revisted_activations
    )
    probability_vector = torch.cat(
        [unknowness_class_prob[:, None], revisted_activations], dim=1
    )

    # Line 7
    probability_vector = F.softmax(probability_vector, dim=1)
    # Line 8
    prediction_score, predicted_class = torch.max(probability_vector, dim=1)
    # Line 9
    prediction_score[predicted_class == 0] = -1.0
    predicted_class = predicted_class - 1

    return predicted_class, prediction_score
