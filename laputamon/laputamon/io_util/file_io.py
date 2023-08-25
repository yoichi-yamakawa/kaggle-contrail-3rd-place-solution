from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Type, Union

import dacite
import fire
import yaml


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    optimizer_params: Optional[Mapping[str, Any]] = field(default_factory=lambda: {"lr": 1e-3})


@dataclass
class TrainingConfig:
    exp_name: str = "sample"
    gpus: int = 1
    n_folds: int = 5
    fold_index: int = 0
    debug: bool = True
    target_name: str = "target"
    max_epochs: int = 2
    batch_size: int = 8
    precision: int = 16
    resume_from_checkpoint: Optional[str] = None
    train_dataset_dir: str = "train_dataset_dir"
    test_dataset_dir: str = "train_dataset_dir"
    optimizer_config: OptimizerConfig = OptimizerConfig()
    logger: object = None
    monitor: str = "val_loss"


def _as_annotated_type(value: str, annotated_type: Type[Any]) -> Any:
    if hasattr(annotated_type, "__origin__") and getattr(annotated_type, "__origin__") is Union:
        for t in annotated_type.__args__:  # type: ignore
            if isinstance(None, t):  # check if t is NoneType
                continue
            try:
                return _as_annotated_type(value, t)
            except ValueError:
                pass
        raise ValueError(f"`{value}` could not be interpreted as `{annotated_type}`")
    return annotated_type(value)


def load_yaml_as_dataclass(target_class: Type[Any], filepath: str, overrides: Optional[Sequence[str]] = None) -> Any:
    if filepath is not None:
        with open(str(filepath), "r") as fp:
            data = yaml.safe_load(fp)
        result = dacite.from_dict(data_class=target_class, data=data)
        print(f"loaded from {filepath}")
        print(result)
    else:
        result = target_class()

    if overrides is not None:
        for keys_value in overrides:
            cat_keys, value = keys_value.split("=")
            lhs = result
            keys = cat_keys.split(".")
            for key in keys[:-1]:
                lhs = getattr(lhs, key)
                assert lhs is not None
            key = keys[-1]

            annotated_type: Optional[Type] = None
            if hasattr(type(lhs), "__annotations__"):
                annotations = getattr(type(lhs), "__annotations__")
                if key in annotations:
                    annotated_type = annotations[key]

            if annotated_type is None:
                annotated_type = type(getattr(lhs, key))

            print(f'override "{key}": {getattr(lhs, key)} -> {value}')

            if value in ["False", "True"]:
                setattr(lhs, key, eval(value))
            else:
                setattr(lhs, key, _as_annotated_type(value, annotated_type))
    return result


def _test_load_yaml(*overrides, target_class=TrainingConfig, filepath="./sample.yaml"):
    result = load_yaml_as_dataclass(target_class, filepath, overrides)

    return result


if __name__ == "__main__":
    fire.Fire(_test_load_yaml)
