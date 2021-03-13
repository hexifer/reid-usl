_base_ = '../_base_/default_runtime.py'

model = dict(
    type='SimSiam',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        strides=(1, 2, 2, 1),
        norm_cfg=dict(type='BN'),
        norm_eval=False),
    neck=dict(
        type='BNNeck',
        feat_dim=2048,
        norm_cfg=dict(type='BN1d'),
        with_bias=False,
        with_avg_pool=True,
        avgpool=dict(type='AvgPoolNeck')),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearPredictor',
            in_channels=2048,
            hid_channels=512,
            out_channels=2048)))

data_source = dict(type='Market1501', data_root='/data/datasets/market1501')
dataset_type = 'ContrastiveDataset'
train_pipeline = [
    dict(
        type='RandomResizedCrop',
        size=(256, 128),
        scale=(0.64, 1.0),
        ratio=(0.33, 0.5),
        interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=10),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(
        type='RandomApply',
        transforms=[dict(type='GaussianBlur', sigma=(0.1, 2.0))],
        p=0.5),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='RandomErasing', value=[0.485, 0.456, 0.406])
]
test_pipeline = [
    dict(type='Resize', size=(256, 128), interpolation=3),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type, data_source=data_source, pipeline=train_pipeline),
    test=dict(
        type='ReIDDataset',
        data_source=data_source,
        pipeline=test_pipeline,
        test_mode=True))

custom_hooks = [dict(type='SetEpochHook')]
paramwise_cfg = {'backbone': dict(lr_mult=0.1)}
optimizer = dict(
    type='SGD',
    lr=0.1,
    weight_decay=0.0001,
    momentum=0.9,
    paramwise_cfg=paramwise_cfg)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.0001,  # cannot be 0
    warmup_by_epoch=True)
log_config = dict(interval=20)
total_epochs = 100
