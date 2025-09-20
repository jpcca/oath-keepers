from config.config import REPO_ROOT
from deeponto.onto import Ontology


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


def get_ontology_info(onto: Ontology, iri: str):
    """
    Ontology インスタンスと IRI から、その IRI の短縮 ID、名前、定義を取得する。

    Arguments:
        onto: owl ファイルをロードした Ontology インスタンス
        iri: "http://purl.obolibrary.org/obo/DOID_0002116" のような完全 IRI

    Returns:
        (short_id, name, definition)
    """
    obj = onto.get_owl_object(iri)
    properties = onto.owl_annotation_properties

    if OwlIri.OBO_ID in properties:
        id_annots = onto.get_annotations(obj, annotation_property_iri=OwlIri.OBO_ID)
        short_id = next(iter(id_annots), None)
    else:
        short_id = None

    if OwlIri.RDFS_LABEL in properties:
        name_annots = onto.get_annotations(obj, annotation_property_iri=OwlIri.RDFS_LABEL)
        name = next(iter(name_annots), None)
    else:
        name = None

    if OwlIri.SYN_EXACT in properties:
        synonym_exact_annots = onto.get_annotations(obj, annotation_property_iri=OwlIri.SYN_EXACT)
        synonym_exact = next(iter(synonym_exact_annots), None)
    else:
        synonym_exact = None

    if OwlIri.SYN_RELATED in properties:
        synonym_related_annots = onto.get_annotations(
            obj, annotation_property_iri=OwlIri.SYN_RELATED
        )
        synonym_related = next(iter(synonym_related_annots), None)
    else:
        synonym_related = None

    if OwlIri.IAO_DEF in properties:
        def_annots = onto.get_annotations(obj, annotation_property_iri=OwlIri.IAO_DEF)
        definition = next(iter(def_annots), None)
    else:
        definition = None

    return (iri, short_id, name, synonym_exact, synonym_related, definition)


if __name__ == "__main__":
    # Ontology のロード
    doid_onto = Ontology(f"{REPO_ROOT}/data/owl/DOID.owl")
    symp_onto = Ontology(f"{REPO_ROOT}/data/owl/SYMP.owl")
    fma_onto = Ontology(f"{REPO_ROOT}/data/owl/FMA.owl")

    # 必要な Ontology に絞る
    doid_list = [
        iri
        for iri in doid_onto.owl_classes.keys()
        if iri.startswith("http://purl.obolibrary.org/obo/DOID_")
    ]
    symp_list = [
        iri
        for iri in symp_onto.owl_classes.keys()
        if iri.startswith("http://purl.obolibrary.org/obo/SYMP_")
    ]
    fma_list = [
        iri
        for iri in fma_onto.owl_classes.keys()
        if iri.startswith("http://purl.obolibrary.org/obo/FMA_")
    ]
    print("len(doid_list) =", len(doid_list))
    print("len(symp_list) =", len(symp_list))
    print("len(fma_list) =", len(fma_list))

    doid_master = [get_ontology_info(doid_onto, doid_iri) for doid_iri in doid_list]
    symp_master = [get_ontology_info(symp_onto, symp_iri) for symp_iri in symp_list]
    fma_master = [get_ontology_info(fma_onto, fma_iri) for fma_iri in fma_list]

    # 出力
    for onto_name, master in [("DOID", doid_master), ("SYMP", symp_master), ("FMA", fma_master)]:
        with open(f"{REPO_ROOT}/data/master/{onto_name}.tsv", "w") as fw:
            fw.write("iri\tid\tname\texact_synonym\trelated_synonym\tdefinition\n")
            for iri, short_id, name, synonym_exact, synonym_related, definition in master:
                fw.write(
                    f"{iri}\t{short_id}\t{name}\t{synonym_exact}\t{synonym_related}\t{definition}\n"
                )

        print(f"{onto_name}_master.tsv saved.")
