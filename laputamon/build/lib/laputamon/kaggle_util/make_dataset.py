import json
import os
import shutil
import subprocess

import fire
import torch


def make_model_dataset_for_kaggle_upload(model_dir, dataset_name, folds=None):
    def _search_best_model(model_dir):
        last_ckpt = model_dir + "/last.ckpt"
        last_ckpt = torch.load(last_ckpt)["callbacks"]

        for k, v in last_ckpt.items():
            if "best_model_path" in v:
                best_model_path = v["best_model_path"]

        return best_model_path

    # make data
    _dataset_name = f"upload_data/{dataset_name}"
    # clean up upload directory
    if os.path.exists(_dataset_name):
        shutil.rmtree(_dataset_name)

    os.makedirs(_dataset_name, exist_ok=True)

    if folds is not None:
        for fold in folds:
            fold_model_dir = f"{model_dir}/{fold}"
            best_model_path = _search_best_model(fold_model_dir)
            # model_dir = os.path.dirname(best_model_path)
            print(f"best model path -> {best_model_path}")
            ckpt = dict()
            ckpt["state_dict"] = torch.load(best_model_path)["state_dict"]

            torch.save(ckpt, f"{_dataset_name}/best-ckpt{fold}.ckpt")
    else:
        best_model_path = _search_best_model(model_dir)
        print(f"best model path -> {best_model_path}")
        ckpt = dict()
        ckpt["state_dict"] = torch.load(best_model_path)["state_dict"]

        torch.save(ckpt, f"{_dataset_name}/best-ckpt.ckpt")

    return _dataset_name


def make_kaggle_model_dataset(model_dir, kaggle_data_name, folds=None):
    dataset_dir = make_model_dataset_for_kaggle_upload(model_dir, "kaggle-dataset", folds)
    upload_to_kaggle(dataset_dir, kaggle_data_name)


def upload_to_kaggle(
    org_dir: str,
    dataset_name: str,
    comments: str = "",
    update: bool = False,
    logger=None,
    extension=".csv",
    subtitle="",
    description="",
    isPrivate=True,
    licenses="unknown",
    keywords=[],
    collaborators=[],
    is_version_update=False,
    delete_old_version=True,
):
    """
    >> upload_to_kaggle(title, k_id, path,  comments, update)

    Arguments
    =========
     title: the title of your dataset.
     k_id: kaggle account id.
     path: non-default string argument of the file path of the data to be uploaded.
     comments:non-default string argument of the comment or the version about your upload.
     logger: logger object if you use logging, default is None.
     extension: the file extension of model weight files, default is ".pth"
     subtitle: the subtitle of your dataset, default is empty string.
     description: dataset description, default is empty string.
     isPrivate: boolean to show wheather to make the data public, default is True.
     licenses = the licenses description, default is "unkown"; must be one of /
     ['CC0-1.0', 'CC-BY-SA-4.0', 'GPL-2.0', 'ODbL-1.0', 'CC-BY-NC-SA-4.0', 'unknown', 'DbCL-1.0', 'CC-BY-SA-3.0', 'copyright-authors', 'other', 'reddit-api', 'world-bank'] .
     keywords : the list of keywords about the dataset, default is empty list.
     collaborators: the list of dataset collaborators, default is empty list.
    """

    k_id = "yoichi7yamakawa"

    data_json = {
        "title": f"{dataset_name[:49]}",
        "id": f"{k_id}/{dataset_name}"[:49],
        "subtitle": subtitle,
        "description": description,
        "isPrivate": isPrivate,
        "licenses": [{"name": licenses}],
        "keywords": [],
        "collaborators": [],
        "data": [],
    }

    with open(org_dir + "/dataset-metadata.json", "w") as f:
        json.dump(data_json, f)

    if not is_version_update:
        create_command = ["kaggle", "datasets", "create", "-p", f"{org_dir}"]
        print(create_command)
        print(subprocess.check_output(create_command))
    else:
        create_command = ["kaggle", "datasets", "version", "-p", f"{org_dir}", "-m", f"{comments}"]
        if delete_old_version:
            create_command.append("-d")

        print(create_command)
        print(subprocess.check_output(create_command))


if __name__ == "__main__":
    fire.Fire({"model_upload": make_kaggle_model_dataset, "simple_upload": upload_to_kaggle})
