from dataclasses import Field, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union


@dataclass
class DatasetConfig:
    name: str = "NFLDatasetV1"
    params: Optional[Dict[str, Any]] = None
    generate_target_online: bool = False


@dataclass
class SlimTrainConfig:
    name: str = "slim_train_v1"
    params: Optional[Dict[str, Any]] = None


@dataclass
class SamplerConfig:
    name: str = "HappyWhaleSamplerV1"
    num_instances: int = 2


@dataclass
class ModelConfig:
    name: str = "Eff_V1"
    gradient_clip_val: float = 1.0
    params: Optional[Any] = field(default_factory=lambda: {})
    layer_freeze_params: Optional[Dict[str, Any]] = None


@dataclass
class SWAConfig:
    swa_epoch_start: float = 5


@dataclass
class LossConfig:
    name: str = "MSELoss"
    params: Optional[Dict[str, Any]] = None


@dataclass
class SecondLossConfig:
    name: str = "MSELoss"
    params: Optional[Dict[str, Any]] = None


@dataclass
class LRConfig:
    name: str = "ExponentialLR"
    interval: str = "epoch"
    lr_scheduler_params: Optional[Any] = field(default_factory=lambda: {"gamma": 0.95})


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    optimizer_params: Optional[Any] = field(default_factory=lambda: {"lr": 1e-3})
    layer_optimize_strategy: Optional[Dict[Any, Any]] = None


@dataclass
class TransformConfig:
    name: str = "sample"
    train_transform: Optional[Any] = field(default_factory=lambda: {"name": "BaseTransformV1"})
    valid_transform: Optional[Any] = field(default_factory=lambda: {"name": "BaseTransformV1"})
    test_transform: Optional[Any] = field(default_factory=lambda: {"name": "BaseTransformV1"})


@dataclass
class TrainingConfig:
    # general settings
    exp_type: str = "starter"
    exp_name: str = "sample"
    debug: bool = True
    log_every_n_steps: int = 50
    val_check_interval: Optional[float] = 1.0
    reload_dataloaders_every_n_epochs: int = 0
    max_epoch_using_pseudo_label: int = 1000
    sampling_frame4: bool = False
    accumulate_grad_batches: int = 1
    train_on_pseudo: bool = False
    scs_dataset_path: Optional[str] = None
    scs_dataset_dir: Optional[str] = None
    scs_mask_dataset_path: Optional[str] = None
    train_on_multi_frames: bool = False
    valid_on_multi_frames: bool = False
    add_meta_data: bool = False
    save_pseudo_label: bool = False
    train_on_all_data: bool = False
    remove_duplicate_train_samples: bool = False
    pred_on_all_frame: bool = False

    # phase: str = "train"
    n_folds: int = 5
    train_on_train_n_folds: bool = False
    fold_index: int = 0
    max_epochs: int = 5

    # cpu/gpu
    training_strategy: str = "auto"
    sync_batchnorm: bool = True
    gpus: int = 2
    precision: Union[str, int] = "16-mixed"
    num_workers: int = 4

    # dataset
    target_name: str = "target"
    train_batch_size: int = 8
    valid_batch_size: int = 32
    test_batch_size: int = 32
    training_type: Optional[str] = None
    train_dataset_dir: str = "p20-contrail/train/"
    pseudo_label_dir: str = "p20-contrail/pseudo_label/exp050-resnest269e-noaug-512-6ch-ind/"
    train_image_dir: str = "p20-contrail/train"
    valid_image_dir: str = "p20-contrail/validation"

    test_dataset_dir: str = "p20-contrail/test/"
    test_image_dir: str = "p20-contrail/test/"

    dataset_config: DatasetConfig = DatasetConfig()
    sampler_config: Optional[SamplerConfig] = None
    train_collate_fn: Optional[str] = None
    valid_collate_fn: Optional[str] = None
    test_collate_fn: Optional[str] = None

    # transform
    transform_config: TransformConfig = TransformConfig()

    # model
    stochastic_weight_averaging: bool = False
    swa_config: SWAConfig = SWAConfig()
    pl_module: str = "BasePLM"

    resume_from_checkpoint: Optional[str] = None
    every_n_epochs: Optional[int] = None
    restart_checkpoint: Optional[str] = None
    model_dir: Optional[str] = None
    ckpt_path: Optional[str] = None
    model_config: ModelConfig = ModelConfig()
    default_model_path: Optional[str] = None
    monitor: Optional[str] = None
    save_top_k: int = 1

    # loss
    loss_config: LossConfig = LossConfig()

    # lr scheduler
    lr_scheduler_config: LRConfig = LRConfig()

    # optimizer
    optimizer_config: OptimizerConfig = OptimizerConfig()
    weight_decay: float = 0.0001

    # logger
    logger: str = "wandb"


if __name__ == "__main__":
    config = TrainingConfig()
    print(config)
