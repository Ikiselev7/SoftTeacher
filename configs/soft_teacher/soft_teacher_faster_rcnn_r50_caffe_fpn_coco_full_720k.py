_base_="base.py"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val=dict(
        ann_file='/content/val/val.json',
        img_prefix='/content/val/images/',),
    test=dict(
        ann_file='/content/test/test.json',
        img_prefix='/content/test/images/',),
    train=dict(

        sup=dict(

            ann_file='/content/train/train.json',
            img_prefix='/content/train/images/',

        ),
        unsup=dict(

            ann_file='/content/unsup/unsup.json',
            img_prefix='/content/unsup/images/',

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

