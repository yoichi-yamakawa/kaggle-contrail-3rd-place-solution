import datetime
from pathlib import Path

import pytz
import wandb


def get_wandb_resume_id(out: str, resume_from_id: bool = False) -> None:
    run_id_path = Path(out) / "wandb_run_id"

    if resume_from_id and run_id_path.exists():
        # resume
        with run_id_path.open("r") as fp:
            run_id = fp.read().strip()
    else:
        # new run
        run_id = wandb.util.generate_id()
        with run_id_path.open("w") as fp:
            fp.write(run_id)

    return run_id


def get_wandb_names(config):
    now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    exp_dt = now.strftime("%m%d-%H%M")
    exp_name_dt = f"{config.exp_name}-fold{config.fold_index}-{exp_dt}"

    return exp_name_dt
