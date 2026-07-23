_base_ = ["./semseg-pt-v3m1-0-base.py"]

# One forward-only pass over the complete training loader.
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
)

# CheckpointLoader must run before ModelHook so fitting uses pretrained weights.
hooks = [
    dict(type="CheckpointLoader", strict=True),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
]
