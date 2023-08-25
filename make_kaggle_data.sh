sh build_zip.sh

# first upload
# python ./laputamon/laputamon/kaggle_util/make_dataset.py \
# simple_upload comp_source contrail-comp-source # first upload 

## For version update of existing dataset (use only after first upload)
python ./laputamon/laputamon/kaggle_util/make_dataset.py \
simple_upload --org_dir=comp_source --dataset_name=contrail-comp-source --is_version_update=True \
--delete_old_version=True \
--comments="update" # version update
