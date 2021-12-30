_base_="base.py"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val=dict(
        ann_file='/content/drive/MyDrive/sartorius/val_dataset_1.json',
        img_prefix='/content/train/',),
    test=dict(
        ann_file='/content/drive/MyDrive/sartorius/val_dataset_1.json',
        img_prefix='/content/train/',),
    train=dict(

        sup=dict(

            ann_file='/content/drive/MyDrive/sartorius/train_dataset_1.json',
            img_prefix='/content/train/',

        ),
        unsup=dict(

            ann_file='/content/drive/MyDrive/sartorius/unsup_dataset_1.json',
            img_prefix='/content/train_semi_supervised/',

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
            epoch_length=2500,
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

evaluation = dict(type="SubModulesDistEvalHook", interval=2500)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(by_epoch=False, interval=2500, max_keep_ckpts=20)
lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

