model = dict(
    type='SoftTeacher',
    model=dict(
        type='MaskRCNN',
        backbone=dict(
            type='SwinTransformer',
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True),
        neck=dict(
            type='FPN',
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            num_outs=3,
            end_level=3),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[2, 5],
                ratios=[0.5, 1.0, 2.0, 3.0],
                strides=[4, 8, 16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='BalancedL1Loss',
                    alpha=0.5,
                    gamma=1.5,
                    beta=1.0,
                    loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=3,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                class_agnostic=False)),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D', dtype='fp16')),
                sampler=dict(
                    type='RandomSampler',
                    num=1024,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=4000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0,
                nms_post=4000),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps2D', dtype='fp16'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=4096,
                    pos_fraction=0.25,
                    add_gt_as_proposals=False,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)),
                mask_size=28,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.15,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=1000,
                mask_thr_binary=0.5))),
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.3,
        rpn_pseudo_threshold=0.5,
        cls_pseudo_threshold=0.5,
        reg_pseudo_threshold=0.01,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=2.0),
    test_cfg=dict(inference_on='student'))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='SemiDataset',
        sup=dict(
            type='CocoDataset',
            ann_file='/content/drive/MyDrive/sartorius/train_dataset_1.json',
            img_prefix='/content/train/',
            classes=['shsy5y', 'cort', 'astro'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='AlbuSegm',
                    transforms=[
                        dict(type='Rotate', limit=45, p=0.2),
                        dict(
                            type='RandomResizedCrop',
                            height=520,
                            width=704,
                            scale=(0.9, 1),
                            p=0.3),
                        dict(type='Flip', p=0.2),
                        dict(type='RandomRotate90', p=0.2),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(
                                    type='ShiftScaleRotate',
                                    shift_limit=0.15,
                                    scale_limit=0.4,
                                    rotate_limit=45)
                            ],
                            p=0.2),
                        dict(type='CLAHE', clip_limit=(1, 8), p=0.2),
                        dict(type='Equalize', p=0.2),
                        dict(
                            type='RandomBrightnessContrast',
                            brightness_limit=[0.1, 0.3],
                            contrast_limit=[0.1, 0.3],
                            p=0.2),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(
                                    type='RGBShift',
                                    r_shift_limit=10,
                                    g_shift_limit=10,
                                    b_shift_limit=10,
                                    p=1.0),
                                dict(
                                    type='HueSaturationValue',
                                    hue_shift_limit=20,
                                    sat_shift_limit=30,
                                    val_shift_limit=20,
                                    p=1.0)
                            ],
                            p=0.1)
                    ],
                    keymap=dict(img='image', gt_masks='masks'),
                    update_pad_shape=False,
                    skip_img_without_anno=True),
                dict(
                    type='Normalize',
                    mean=[128, 128, 128],
                    std=[11.58, 11.58, 11.58],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ExtraAttrs', tag='sup'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'img_norm_cfg', 'pad_shape', 'scale_factor',
                               'tag'))
            ]),
        unsup=dict(
            type='CocoDataset',
            ann_file='/content/drive/MyDrive/sartorius/unsup_dataset_1.json',
            img_prefix='/content/train_semi_supervised/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='PseudoSamples', with_bbox=True, with_mask=True),
                dict(
                    type='AlbuSegm',
                    transforms=[
                        dict(type='Rotate', limit=45, p=0.2),
                        dict(
                            type='RandomResizedCrop',
                            height=520,
                            width=704,
                            scale=(0.9, 1),
                            p=0.3),
                        dict(type='Flip', p=0.2),
                        dict(type='RandomRotate90', p=0.2),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(
                                    type='ShiftScaleRotate',
                                    shift_limit=0.15,
                                    scale_limit=0.4,
                                    rotate_limit=45)
                            ],
                            p=0.2),
                        dict(type='CLAHE', clip_limit=(1, 8), p=0.2),
                        dict(type='Equalize', p=0.2),
                        dict(
                            type='RandomBrightnessContrast',
                            brightness_limit=[0.1, 0.3],
                            contrast_limit=[0.1, 0.3],
                            p=0.2),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(
                                    type='RGBShift',
                                    r_shift_limit=10,
                                    g_shift_limit=10,
                                    b_shift_limit=10,
                                    p=1.0),
                                dict(
                                    type='HueSaturationValue',
                                    hue_shift_limit=20,
                                    sat_shift_limit=30,
                                    val_shift_limit=20,
                                    p=1.0)
                            ],
                            p=0.1)
                    ],
                    keymap=dict(img='image', gt_masks='masks'),
                    update_pad_shape=False,
                    skip_img_without_anno=True),
                dict(
                    type='MultiBranch',
                    unsup_student=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(704, 520), (704, 520)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5),
                                dict(
                                    type='ShuffledSequential',
                                    transforms=[
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(type='Identity'),
                                                dict(type='AutoContrast'),
                                                dict(type='RandEqualize'),
                                                dict(type='RandSolarize'),
                                                dict(type='RandColor'),
                                                dict(type='RandContrast'),
                                                dict(type='RandBrightness'),
                                                dict(type='RandSharpness'),
                                                dict(type='RandPosterize')
                                            ]),
                                        dict(
                                            type='OneOf',
                                            transforms=[{
                                                'type': 'RandTranslate',
                                                'x': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandTranslate',
                                                'y': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandRotate',
                                                'angle': (-30, 30)
                                            },
                                                [{
                                                    'type':
                                                        'RandShear',
                                                    'x': (-30, 30)
                                                }, {
                                                    'type':
                                                        'RandShear',
                                                    'y': (-30, 30)
                                                }]])
                                    ]),
                                dict(
                                    type='RandErase',
                                    n_iterations=(1, 5),
                                    size=[0, 0.2],
                                    squared=True)
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[128, 128, 128],
                            std=[11.58, 11.58, 11.58],
                            to_rgb=True),
                        dict(type='ExtraAttrs', tag='unsup_student'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ],
                    unsup_teacher=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(704, 520), (704, 520)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5)
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[128, 128, 128],
                            std=[11.58, 11.58, 11.58],
                            to_rgb=True),
                        dict(type='ExtraAttrs', tag='unsup_teacher'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ])
            ],
            classes=['shsy5y', 'cort', 'astro'],
            filter_empty_gt=False)),
    val=dict(
        type='CocoDataset',
        ann_file='/content/drive/MyDrive/sartorius/val_dataset_1.json',
        img_prefix='/content/train/',
        classes=['shsy5y', 'cort', 'astro'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(704, 520), (704, 520)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[128, 128, 128],
                        std=[11.58, 11.58, 11.58],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='/content/drive/MyDrive/sartorius/val_dataset_1.json',
        img_prefix='/content/train/',
        classes=['shsy5y', 'cort', 'astro'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(704, 520), (704, 520)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[128, 128, 128],
                        std=[11.58, 11.58, 11.58],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    sampler=dict(
        train=dict(
            type='SemiBalanceSampler',
            sample_ratio=[1, 1],
            by_prob=True,
            epoch_length=2500)))
evaluation = dict(interval=2500, metric=['bbox', 'segm'], type='SubModulesDistEvalHook', classwise=True)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[480000, 640000])
runner = dict(type='IterBasedRunner', max_iters=720000)
checkpoint_config = dict(interval=2500, by_epoch=False, max_keep_ckpts=20)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='pre_release',
                name='swin_mask',
                config=dict(
                    work_dirs=
                    './work_dirs/swin_mask',
                    total_step=720000)),
            by_epoch=False)
    ])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='WeightSummary'),
    dict(type='MeanTeacher', momentum=0.999, interval=1, warm_up=0)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/content/drive/MyDrive/sartorius/swin_base_coco_cats_iou_s/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]
mmdet_base = '../../thirdparty/mmdetection/configs/_base_'
fp16 = dict(loss_scale='dynamic')
work_dir = './work_dirs/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k'
cfg_name = 'soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k'
gpu_ids = range(0, 1)
find_unused_parameters=True