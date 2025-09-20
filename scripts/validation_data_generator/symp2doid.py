import builtins
import os
from typing import List

# for JVM memory setting, this must be set before importing Ontology
builtins.input = lambda _: os.environ.get("JVM_MAX_MEMORY", "4g")

import numpy as np
import pandas as pd
import tqdm
from config.config import REPO_ROOT, OwlIri
from deeponto.onto import Ontology
from org.semanticweb.owlapi.model import OWLObjectSomeValuesFrom
from pandas import DataFrame
from rapidfuzz.process import cdist
from sentence_transformers import SentenceTransformer, util


def join_info(df_result, df_doid, df_symp):
    return (
        pd.merge(
            pd.merge(
                df_result,
                df_doid[["iri", "name"]].rename({"name": "disease_name", "iri": "doid_iri"}, axis=1),
                on="doid_iri",
                how="inner",
            ),
            df_symp[["iri", "name", "definition"]].rename(
                {"name": "symptom_name", "definition": "symptom_definition", "iri": "symp_iri"}, axis=1
            ),
            on="symp_iri",
            how="inner",
        )
        .sort_values(["symp_iri", "doid_iri"])
    )[["symp_iri", "doid_iri", "disease_name", "symptom_name", "symptom_definition", "method", "score"]]


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
        Dict of doid_iri -> List of tuples (symp_iri, method, score)
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

    return pd.DataFrame(result, columns=["doid_iri", "symp_iri", "method", "score"])


def symp2doid_by_sentence(df_symp: DataFrame, df_doid: DataFrame, logic: str, topk: int = 5):
    """
    symp と doid の名前、定義、同義語を使ったキーワードベースのマッチング
    e.g.
        DOID:0050117  Malaria  hasExactSynonym: "malaria"
        SYMP:0000185 Fever   name: "fever", hasExactSynonym: "pyrexia"

    Arguments:
        df_symp: DataFrame for SYMP.tsv
        df_doid: DataFrame for DOID.tsv
        logic: "levenshtein" or "cosine"
        topk: Return top k results for each doid
    Returns:
        Dict of doid_iri -> List of tuples (symp_iri, method)
        where method is "synonym"
    """
    results = []
    _df_symp = df_symp.copy().fillna("")
    _df_doid = df_doid.copy().fillna("")

    def _join_keywords(row):
        return f"name={row['name']}:exact_synonym={row['exact_synonym']}:related_synonym={row['related_synonym']}:definition={row['definition']}"

    _df_symp["sentence"] = (
        _df_symp[["name", "exact_synonym", "related_synonym", "definition"]]
        .apply(_join_keywords, axis=1)
        .str.lower()
    )
    _df_doid["sentence"] = (
        _df_doid[["name", "exact_synonym", "related_synonym", "definition"]]
        .apply(_join_keywords, axis=1)
        .str.lower()
    )

    symp_sentence = _df_symp["sentence"].tolist()
    doid_sentence = _df_doid["sentence"].tolist()
    rows = np.arange(len(_df_symp))[:, None]

    def _get_topk(logic=logic, topk=topk):
        if logic == "levenshtein":
            scores = cdist(symp_sentence, doid_sentence, workers=-1) / 100
        elif logic == "cosine":
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            symp_embeddings = model.encode(
                symp_sentence, convert_to_tensor=True, show_progress_bar=True
            )
            doid_embeddings = model.encode(
                doid_sentence, convert_to_tensor=True, show_progress_bar=True
            )
            scores = util.cos_sim(symp_embeddings, doid_embeddings).cpu().numpy()

        topk_indices = np.argsort(-scores, axis=1)[:, :topk]
        topk_doid_iris = [
            _df_doid.iloc[topk_index]["iri"].tolist() for topk_index in topk_indices
        ]  # shape: (num_symptoms, topk)
        topk_scores = scores[rows, topk_indices].tolist()  # shape: (num_symptoms, topk)

        return (topk_doid_iris, topk_scores)

    topk_doid_iris, topk_scores = _get_topk(logic, topk)
    for symp_iri, topk_doid_iri, topk_score in zip(_df_symp["iri"], topk_doid_iris, topk_scores):
        for doid_iri, confidence in zip(topk_doid_iri, topk_score):
            results.append([doid_iri, symp_iri, logic, confidence])

    return pd.DataFrame(results, columns=["doid_iri", "symp_iri", "method", "score"])


if __name__ == "__main__":
    doid_onto = Ontology(f"{REPO_ROOT}/data/owl/DOID.owl")
    df_symp = pd.read_table(f"{REPO_ROOT}/data/master/SYMP.tsv", sep="\t", header=0, dtype=str)
    df_doid = pd.read_table(f"{REPO_ROOT}/data/master/DOID.tsv", sep="\t", header=0, dtype=str)
    doid_list = df_doid["iri"].tolist()

    # Run axiom-based matching
    print("Running axiom-based matching...")
    df_result_axiom = doid2symp_by_axiom(doid_onto, doid_list)
    df_result_axiom = join_info(df_result_axiom, df_doid, df_symp)
    df_result_axiom.to_csv(f"{REPO_ROOT}/data/relationship/symp2doid_by_axiom.tsv", sep="\t", index=False)

    # Run levenshtein-based matching
    print("Running levenshtein-based matching...")
    df_result_levenshtein = symp2doid_by_sentence(df_symp, df_doid, logic="levenshtein", topk=5)
    df_results_levenshtein = join_info(df_result_levenshtein, df_doid, df_symp)
    df_results_levenshtein.to_csv(f"{REPO_ROOT}/data/relationship/symp2doid_by_keyword.tsv", sep="\t", index=False)

    # Run cosine-based matching
    print("Running cosine-based matching...")
    df_result_semantic = symp2doid_by_sentence(df_symp, df_doid, logic="cosine", topk=5)
    df_result_semantic = join_info(df_result_semantic, df_doid, df_symp)
    df_result_semantic.to_csv(f"{REPO_ROOT}/data/relationship/symp2doid_by_semantic.tsv", sep="\t", index=False)

    print("complete!")
