import subprocess

REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
).stdout.strip()


# アノテーションプロパティの IRI
class OwlIri:
    RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"  # label = その IRI の名前を表す IRI
    OBO_ID = (
        "http://www.geneontology.org/formats/oboInOwl#id"  # OBO ID = その IRI の短縮 ID を表す IRI
    )
    IAO_DEF = "http://purl.obolibrary.org/obo/IAO_0000115"  # IAO = その IRI の定義を表す IRI
    SYN_EXACT = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"  # 同義語 (厳密)
    SYN_RELATED = "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"  # 関連語
    RO_SYMPTOM = "RO_0002452"  # this means `doid has_symptom symp` https://www.ebi.ac.uk/ols4/ontologies/ro/properties/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FRO_0002452
