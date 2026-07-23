import json
from pathlib import Path

import torch

import pointcept.utils.comm as comm
from pointcept.engines.hooks import HookBase
from pointcept.models.builder import MODELS
from pointcept.models.default import DefaultSegmentorV2
from pointcept.models.utils.structure import Point

from .openmax import (
    apply_openmax,
    assemble_openmax_models,
    collect_correct_activations,
    fit_openmax_class,
    load_openmax_models,
    save_openmax_models,
)


@MODELS.register_module()
class OpenMaxSegmentor(DefaultSegmentorV2, HookBase):
    """
    DefaultSegmentorV2 with fitted OpenMax recalibration at inference time.

    Its learnable parameter names and shapes are identical to
    ``DefaultSegmentorV2``, so a pretrained segmentor checkpoint can be loaded
    strictly.  The original closed-set logits remain available under
    ``closed_set_logits``; ``seg_logits`` contains the alpha-reweighted known
    class activations for compatibility with existing semantic-segmentation
    inference code.
    """

    requires_backprop = False

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        openmax_model_path=None,
        openmax_alpha=10,
        run_paper_version=True,
        reweight_seg_logits=True,
        openmax_fit_output=None,
        openmax_activation_dir=None,
        openmax_tailsize=1.0,
        openmax_distance_multiplier=1.0,
        openmax_translate_amount=1.0,
        openmax_distance_metric="cosine",
    ):
        super().__init__(
            num_classes=num_classes,
            backbone_out_channels=backbone_out_channels,
            backbone=backbone,
            criteria=criteria,
            freeze_backbone=freeze_backbone,
        )
        if not isinstance(openmax_alpha, int) or openmax_alpha < 1:
            raise ValueError(
                "openmax_alpha must be a positive integer, "
                f"got {openmax_alpha}"
            )
        self.num_classes = num_classes
        self.openmax_alpha = openmax_alpha
        self.run_paper_version = run_paper_version
        self.reweight_seg_logits = reweight_seg_logits
        self.openmax_fit_output = openmax_fit_output
        self.openmax_activation_dir = openmax_activation_dir
        self.openmax_tailsize = openmax_tailsize
        self.openmax_distance_multiplier = openmax_distance_multiplier
        self.openmax_translate_amount = openmax_translate_amount
        self.openmax_distance_metric = openmax_distance_metric
        self.openmax_models = None
        self._fit_active = False
        self._activation_shards = None
        self._activation_counts = None
        if openmax_model_path is not None:
            self.load_openmax(openmax_model_path)

    def load_openmax(self, path):
        self.openmax_models = load_openmax_models(path)
        metadata = self.openmax_models["metadata"]
        expected_class_ids = list(range(self.num_classes))
        class_ids = [int(class_id) for class_id in metadata["class_ids"]]
        if class_ids != expected_class_ids:
            raise ValueError(
                "OpenMax classes must match segmentor output columns. "
                f"Expected {expected_class_ids}, got {class_ids}"
            )
        if metadata["activation_dim"] != self.num_classes:
            raise ValueError(
                "OpenMax activation dimension must equal num_classes. "
                f"Expected {self.num_classes}, got "
                f"{metadata['activation_dim']}"
            )

    def before_train(self):
        if self.openmax_models is not None:
            raise RuntimeError(
                "Cannot fit OpenMax while openmax_model_path is configured"
            )
        if self.trainer.start_epoch != 0 or self.trainer.cfg.resume:
            raise RuntimeError("OpenMax fitting does not support resume")
        if self.trainer.max_epoch != 1:
            raise ValueError(
                "OpenMax fitting must traverse the training data once. "
                "Set epoch=1 and eval_epoch=1."
            )
        if self.trainer.cfg.data.train.loop != 1:
            raise ValueError(
                "OpenMax fitting requires data.train.loop=1. "
                "Set epoch=1 and eval_epoch=1."
            )
        if self.trainer.cfg.mix_prob != 0:
            raise ValueError("OpenMax fitting requires mix_prob=0")
        if self.trainer.train_loader.drop_last:
            raise ValueError("OpenMax fitting requires drop_last=False")
        if self.trainer.cfg.evaluate:
            raise ValueError("OpenMax fitting requires evaluate=False")

        model_dir = Path(self.trainer.cfg.save_path) / "model"
        if self.openmax_fit_output is None:
            self.openmax_fit_output = model_dir / "openmax_models.pth"
        else:
            self.openmax_fit_output = Path(self.openmax_fit_output)
        if self.openmax_activation_dir is None:
            self.openmax_activation_dir = model_dir / "openmax_activations"
        else:
            self.openmax_activation_dir = Path(self.openmax_activation_dir)
        self.openmax_activation_dir.mkdir(parents=True, exist_ok=True)
        self._fit_active = True
        self.trainer.logger.info(
            "=> OpenMax fitting initialized; collecting correct activations"
        )

    def before_epoch(self):
        self.eval()
        self._activation_shards = {
            class_id: [] for class_id in range(self.num_classes)
        }
        self._activation_counts = {
            class_id: 0 for class_id in range(self.num_classes)
        }

    def _forward_logits(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat(
                    [parent.feat, point.feat[inverse]], dim=-1
                )
                point = parent
            feat = point.feat
        else:
            feat = point

        closed_set_logits = self.seg_head(feat)
        return closed_set_logits, point

    def _collect_activations(self, closed_set_logits, segment):
        batch_activations = collect_correct_activations(
            activations=closed_set_logits,
            segment=segment,
            num_classes=self.num_classes,
        )
        count = sum(
            activations.shape[0]
            for activations in batch_activations.values()
        )
        if batch_activations:
            epoch = self.trainer.epoch
            iteration = self.trainer.comm_info["iter"]
            for class_id, activations in batch_activations.items():
                shard_dir = (
                    self.openmax_activation_dir
                    / f"class_{class_id:04d}"
                    / f"rank_{comm.get_rank():04d}"
                )
                shard_dir.mkdir(parents=True, exist_ok=True)
                shard_path = shard_dir / (
                    f"epoch_{epoch:04d}_batch_{iteration:08d}.pth"
                )
                temporary_path = shard_path.with_suffix(".tmp")
                torch.save(
                    {
                        "version": 1,
                        "class_id": class_id,
                        "rank": comm.get_rank(),
                        "epoch": epoch,
                        "iteration": iteration,
                        "activations": activations,
                    },
                    temporary_path,
                )
                temporary_path.replace(shard_path)
                self._activation_shards[class_id].append(shard_path)
                self._activation_counts[class_id] += activations.shape[0]
        return closed_set_logits.new_tensor(float(count))

    def _load_local_class_activations(self, class_id):
        shard_paths = self._activation_shards[class_id]
        if not shard_paths:
            return torch.empty((0, self.num_classes))

        activations = None
        offset = 0
        for shard_path in shard_paths:
            shard = torch.load(
                shard_path, map_location="cpu", weights_only=True
            )
            if int(shard["class_id"]) != class_id:
                raise ValueError(
                    f"Activation shard '{shard_path}' contains class "
                    f"{shard['class_id']}, expected {class_id}"
                )
            values = shard["activations"]
            if activations is None:
                activations = values.new_empty(
                    (
                        self._activation_counts[class_id],
                        self.num_classes,
                    )
                )
            next_offset = offset + values.shape[0]
            activations[offset:next_offset].copy_(values)
            offset = next_offset
        if offset != activations.shape[0]:
            raise ValueError(
                f"Loaded {offset} activations for class {class_id}, "
                f"expected {activations.shape[0]}"
            )
        return activations

    def _write_activation_manifest(self, rank_shards, rank_counts):
        counts = {
            class_id: sum(
                counts_for_rank[class_id]
                for counts_for_rank in rank_counts
            )
            for class_id in range(self.num_classes)
        }
        manifest_path = self.openmax_activation_dir / "manifest.json"
        temporary_path = manifest_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "num_classes": self.num_classes,
                    "classes": {
                        str(class_id): {
                            "activation_count": counts[class_id],
                            "shards": [
                                shard
                                for shards_for_rank in rank_shards
                                for shard in shards_for_rank[class_id]
                            ],
                        }
                        for class_id in range(self.num_classes)
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
        return counts

    def _merge_class_activations(self, rank_activations):
        values = [
            activations
            for activations in rank_activations
            if activations.shape[0] > 0
        ]
        if len(values) == 1:
            return values[0]
        return torch.cat(values, dim=0)

    def after_epoch(self):
        local_shards = {
            class_id: [
                path.relative_to(self.openmax_activation_dir).as_posix()
                for path in self._activation_shards[class_id]
            ]
            for class_id in range(self.num_classes)
        }
        rank_shards = comm.all_gather(local_shards)
        rank_counts = comm.all_gather(self._activation_counts)
        counts = {
            class_id: sum(
                counts_for_rank[class_id]
                for counts_for_rank in rank_counts
            )
            for class_id in range(self.num_classes)
        }
        if comm.is_main_process():
            self._write_activation_manifest(rank_shards, rank_counts)
            self.trainer.logger.info(
                "Correct OpenMax activation counts per class: %s", counts
            )
            self.trainer.logger.info(
                "=> Saved incremental OpenMax activation shards to '%s'",
                self.openmax_activation_dir,
            )

        missing = [
            class_id for class_id, count in counts.items() if count == 0
        ]
        if missing:
            raise ValueError(
                "Cannot fit OpenMax without correctly classified activations "
                f"for every class. Missing classes: {missing}"
            )

        fitted_classes = {}
        fit_device = next(self.parameters()).device
        for class_id in range(self.num_classes):
            local_activations = self._load_local_class_activations(class_id)
            rank_activations = comm.gather(local_activations, dst=0)
            if comm.is_main_process():
                activations = self._merge_class_activations(
                    rank_activations
                )
                fitted_classes[class_id] = fit_openmax_class(
                    activations=activations,
                    class_id=class_id,
                    tailsize=self.openmax_tailsize,
                    distance_multiplier=self.openmax_distance_multiplier,
                    translate_amount=self.openmax_translate_amount,
                    distance_metric=self.openmax_distance_metric,
                    device=fit_device,
                )
                self.trainer.logger.info(
                    "=> Fitted OpenMax class %d from %d activations",
                    class_id,
                    counts[class_id],
                )

        if comm.is_main_process():
            self.openmax_models = assemble_openmax_models(
                fitted_classes=fitted_classes,
                tailsize=self.openmax_tailsize,
                distance_multiplier=self.openmax_distance_multiplier,
                translate_amount=self.openmax_translate_amount,
                distance_metric=self.openmax_distance_metric,
            )
            save_openmax_models(
                self.openmax_models, self.openmax_fit_output
            )
            self.trainer.logger.info(
                "=> Saved fitted OpenMax models to '%s'",
                self.openmax_fit_output,
            )
        comm.synchronize()
        self._activation_shards = None
        self._activation_counts = None
        self._fit_active = False

    def forward(self, input_dict, return_point=False):
        closed_set_logits, point = self._forward_logits(input_dict)
        if self._fit_active:
            if "segment" not in input_dict:
                raise KeyError(
                    "OpenMax fitting requires ground-truth 'segment' labels"
                )
            correct_count = self._collect_activations(
                closed_set_logits, input_dict["segment"]
            )
            return {"correct_activations": correct_count}

        return_dict = {"seg_logits": closed_set_logits}
        if return_point:
            return_dict["point"] = point
        if self.training or self.openmax_models is None:
            return return_dict

        openmax_result = apply_openmax(
            activations=closed_set_logits,
            fitted_models=self.openmax_models,
            alpha=self.openmax_alpha,
            run_paper_version=self.run_paper_version,
        )
        return_dict["closed_set_logits"] = closed_set_logits
        return_dict["openmax"] = openmax_result
        if self.reweight_seg_logits:
            return_dict["seg_logits"] = openmax_result[
                "revised_activations"
            ]
        return return_dict
