# build package
python -m build

# deploy to PyPI
twine upload dist/*