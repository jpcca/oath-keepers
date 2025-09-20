from typing import List

import numpy as np
import pandas as pd
import tqdm
from config.config import REPO_ROOT, OwlIri
from deeponto.onto import Ontology
from org.semanticweb.owlapi.model import AxiomType, OWLObjectSomeValuesFrom
from pandas import DataFrame
from rapidfuzz.process import cdist


def doid2symp_by_axiom(doid_onto: Ontology, doid_list: List):
    """
    RO_0002452(has_symp) を使った axiom ベースのマッチング
    ontology の axiom では `doid has_symptom symp` という形式の axiom しか存在しないため
    doid から symptom を引く。
    e.g.
        Malaria ⊑ Disease ⊓ has_symptom some Fever

    Arguments:
        doid_onto:  Ontology instance for DOID.owl
    Returns:
        Dict of doid_iri -> List of tuples (symptom_iri, method, score)
        where method is "axiom" and score is 1.0
    """
    result = []

    for doid_iri in tqdm.tqdm(doid_list):
        doid_obj = doid_onto.get_owl_object(doid_iri)
        for sup in doid_onto.get_asserted_parents(doid_obj, named_only=False):
            if isinstance(sup, OWLObjectSomeValuesFrom):
                prop = sup.getProperty()
                if OwlIri.RO_SYMPTOM in str(prop):
                    filler = sup.getFiller()
                    symp_iri = str(filler.getIRI())
                    result.append([doid_iri, symp_iri, "axiom", 1.0])

    return pd.DataFrame(result, columns=["doid_iri", "symptom_iri", "method", "score"])


def symp2doid_by_keyword(df_symp: DataFrame, df_doid: DataFrame, topk=5):
    """
    symp と doid の名前、定義、同義語を使ったキーワードベースのマッチング
    e.g.
        DOID:0050117  Malaria  hasExactSynonym: "malaria"
        SYMP:0000185 Fever   name: "fever", hasExactSynonym: "pyrexia"

    Arguments:
        df_symp: DataFrame for SYMP.tsv
        df_doid: DataFrame for DOID.tsv
        topk: Return top k results for each doid
    Returns:
        Dict of doid_iri -> List of tuples (symptom_iri, method)
        where method is "synonym"
    """
    results = []
    _df_symp = df_symp.copy().fillna("")
    _df_doid = df_doid.copy().fillna("")

    def _join_keywords(row):
        return f"name={row['name']}:exact_synonym={row['exact_synonym']}:related_synonym={row['related_synonym']}:definition={row['definition']}"

    _df_symp["keyword"] = (
        _df_symp[["name", "exact_synonym", "related_synonym", "definition"]]
        .apply(_join_keywords, axis=1)
        .str.lower()
    )
    _df_doid["keyword"] = (
        _df_doid[["name", "exact_synonym", "related_synonym", "definition"]]
        .apply(_join_keywords, axis=1)
        .str.lower()
    )

    queries = _df_symp["keyword"].tolist()
    choices = _df_doid["keyword"].tolist()

    scores = cdist(queries, choices, workers=-1)
    topk_indices = np.argsort(scores, axis=1)[:, -topk:][:, ::-1]
    rows = np.arange(scores.shape[0])[:, None]
    topk_doid_iris = [
        _df_doid.iloc[topk_index]["iri"].tolist() for topk_index in topk_indices
    ]  # shape: (num_symptoms, topk)
    topk_confidences = scores[rows, topk_indices].tolist()  # shape: (num_symptoms, topk)

    for symp_iri, topk_doid_iri, topk_confidence in zip(
        _df_symp["iri"], topk_doid_iris, topk_confidences
    ):
        for doid_iri, confidence in zip(topk_doid_iri, topk_confidence):
            results.append([doid_iri, symp_iri, "keyword", confidence])

    return pd.DataFrame(results, columns=["doid_iri", "symptom_iri", "method", "score"])


if __name__ == "__main__":
    # Run axiom-based matching
    print("Running axiom-based matching...")
    doid_onto = Ontology(f"{REPO_ROOT}/data/owl/DOID.owl")
    df_symp = pd.read_table(f"{REPO_ROOT}/data/master/SYMP.tsv", sep="\t", header=0, dtype=str)
    df_doid = pd.read_table(f"{REPO_ROOT}/data/master/DOID.tsv", sep="\t", header=0, dtype=str)
    doid_list = df_doid["iri"].tolist()

    df_results_axiom = doid2symp_by_axiom(doid_onto, doid_list)
    (
        df_results_axiom.set_index("doid_iri")
        .join(df_doid.set_index("iri")["name"].rename("doid_name"))
        .reset_index()
        .set_index("symptom_iri")
        .join(
            df_symp.set_index("iri")[["name", "definition"]].rename(
                {"name": "symptom_name", "definition": "symptom_definition"}, axis=1
            )
        )
        .reset_index()
        .rename({"index": "symptom_iri"}, axis=1)
    ).to_csv(f"{REPO_ROOT}/data/relationship/symp2doid_by_axiom.tsv", sep="\t", index=False)

    # Run keyword-based matching
    print("Running keyword-based matching...")
    df_results_keyword = symp2doid_by_keyword(df_symp, df_doid, topk=5)
    (
        df_results_keyword.set_index("doid_iri")
        .join(df_doid.set_index("iri")["name"].rename("doid_name"))
        .reset_index()
        .set_index("symptom_iri")
        .join(
            df_symp.set_index("iri")[["name", "definition"]].rename(
                {"name": "symptom_name", "definition": "symptom_definition"}, axis=1
            )
        )
        .reset_index()
        .rename({"index": "symptom_iri"}, axis=1)
    ).to_csv(f"{REPO_ROOT}/data/relationship/symp2doid_by_keyword.tsv", sep="\t", index=False)

    print("complete!")
