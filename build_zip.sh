rm -rf dependencies
rm -f source.zip
rm -rf comp_source

python3 -m pip wheel -w dependencies --no-deps -r requirements_for_notebook.txt
python3 -m pip wheel -w dependencies --no-deps ./laputamon

# rsync -av scripts/ comp_source/ --exclude "*/__pycache__/*"
mkdir comp_source

# cp scrip
zip -r source.zip \
  scripts \
  dependencies \
  -x \*/__pycache__/\* \
  -x \*/.\* \
  -x \*/.\*/\* \
 
mv source.zip comp_source