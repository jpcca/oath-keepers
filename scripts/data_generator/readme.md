## prepare
```shell
uv add rapidfuzz
uv add deeponto
brew install openjdk
# follow the prompt after you install openjdk to export path
```


## deeponto
https://github.com/KRR-Oxford/DeepOnto

- ontology
    - DOID: ontology about desease
    - SYMP: ontology about symptoms of desease
    - FMA: ontology about anatomy
- iri: unique id for the ontology


## logic
### desease and symptom relationship
- two matching logic
    1. as if doid has the relationship `has symptom`
    2. as if doid has the relationship `has synonym`
- a desease has some symptomps
    - desease:symptom = 1:N

### symptom and location relationship
- keyword matching of symptom(`SYMP` ontology) definition and location name in `FMA` ontology.
- a symptom has some locations
    - symptom:location = 1:N