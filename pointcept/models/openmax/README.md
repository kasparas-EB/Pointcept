# OpenMax semantic segmentation

OpenMax fitting uses the segmentor's per-point class logits as activation
vectors. Only points whose closed-set `argmax` equals their ground-truth
`segment` label are retained, grouped by ground-truth class, and used to fit
one mean activation vector (MAV) and Weibull tail model per class.

## Fit from a pretrained segmentor

Create a fitting config that inherits the pretrained segmentor config. The
standard trainer is used for one forward-only epoch; `ModelHook` invokes the
fitting lifecycle implemented by `OpenMaxSegmentor`.

```python
_base_ = ["./semseg-pt-v3m1-0-base.py"]

epoch = 1
eval_epoch = 1
evaluate = False
mix_prob = 0
drop_last = False
enable_wandb = False

model = dict(
    type="OpenMaxSegmentor",
    openmax_tailsize=0.05,
    openmax_distance_multiplier=1.0,
    openmax_translate_amount=1.0,
    openmax_distance_metric="cosine",
    # Defaults under <save_path>/model:
    # openmax_fit_output=".../openmax_models.pth",
    # openmax_activation_dir=".../openmax_activations",
)

# CheckpointLoader must run before ModelHook.
hooks = [
    dict(type="CheckpointLoader", strict=True),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
]
```

Run it through the normal training entry point:

```bash
python tools/train.py \
  --config-file configs/scannet/semseg-pt-v3m1-0-openmax.py \
  --options \
  weight=exp/scannet/semseg-pt-v3m1-0-base/model/model_best.pth \
  save_path=exp/scannet/semseg-pt-v3m1-0-openmax
```

The model validates the one-pass fitting settings in `before_train` and
switches to evaluation behavior in `before_epoch`. During `forward`, every
batch's correctly classified activations are immediately moved to CPU and
written to separate per-class, per-rank shards. Only shard paths and per-class
counts remain in memory. `manifest.json` indexes the shards by class, so a
consumer can load one selected class without reading any other class.
`after_epoch` also fits classes sequentially: it loads, gathers, and fits one
class before moving to the next. No backward pass or optimizer step is
performed. Every class must have at least one correctly classified training
point; fitting reports missing class ids otherwise.

`tailsize <= 1` is interpreted as a fraction of the available distances,
matching the VAST implementation. A value greater than one is interpreted as
an absolute count.

## Run inference

Change only the segmentor type and add the fitted model settings:

```python
model = dict(
    type="OpenMaxSegmentor",
    num_classes=20,
    backbone_out_channels=64,
    backbone=...,
    criteria=...,
    openmax_model_path=(
        "exp/scannet/semseg-pt-v3m1-0-base/model/openmax_models.pth"
    ),
    openmax_alpha=10,
    run_paper_version=True,
)
```

`OpenMaxSegmentor` has the same learnable state as `DefaultSegmentorV2`, so the
original pretrained checkpoint loads strictly. In evaluation/test mode its
output contains:

- `closed_set_logits`: original `[N, C]` activations.
- `seg_logits`: alpha-reweighted `[N, C]` known-class activations, preserving
  compatibility with the existing semantic segmentation tester.
- `openmax`: a datadict containing `[N, C + 1]` `logits` and `probabilities`
  (unknown is column zero), `[N, C]` `evt_knownness`,
  `revised_activations`, `unknown_probability`, and zero-based `prediction`
  (`-1` means unknown).

`OpenMaxSegmentor` is intended for fitting and inference over a pretrained
model. Its forward pass never computes training criteria, even when `segment`
is present in the input.
