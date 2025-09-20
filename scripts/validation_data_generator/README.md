## prepare
install OpenJDK to install Java
```shell
# follow the prompt after you install openjdk to export path
brew install openjdk
```

install python libraries
```shell
uv add rapidfuzz
uv add deeponto
```

set python interpreter
```shell
source .venv/bin/activate
```

## deeponto
https://github.com/KRR-Oxford/DeepOnto

- ontology
    - DOID: ontology about desease
    - SYMP: ontology about symptoms of desease
    - FMA: ontology about anatomy
- iri: unique id for the ontology


## run
```python
python scripts/validation_data_generator/dowloader.py
python scripts/validation_data_generator/master_data_generator.py
python scripts/validation_data_generator/symp2doid.py
```
