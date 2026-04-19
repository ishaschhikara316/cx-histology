from workflow.scripts import gdc_manifest as gm


def test_build_filters_for_cancer_type_returns_dict():
    f = gm.build_filters(project="TCGA-LUAD",
                         data_type="Slide Image",
                         experimental_strategy="Diagnostic Slide",
                         data_format="SVS")
    assert f["op"] == "and"
    ops = {c["content"]["field"]: c["content"]["value"] for c in f["content"]}
    assert ops["cases.project.project_id"] == ["TCGA-LUAD"]
    assert ops["files.data_format"] == ["SVS"]


def test_parse_gdc_response_to_manifest():
    fake = {
        "data": {"hits": [
            {"file_id": "abc", "file_name": "TCGA-02-0001-01Z-00-DX1.svs",
             "file_size": 123, "md5sum": "d41d"},
            {"file_id": "def", "file_name": "TCGA-02-0002-01Z-00-DX1.svs",
             "file_size": 456, "md5sum": "e99a"},
        ]}
    }
    rows = gm.hits_to_manifest(fake["data"]["hits"], project="TCGA-LUAD")
    assert len(rows) == 2
    assert rows[0]["project"] == "TCGA-LUAD"
    assert rows[0]["patient"] == "TCGA-02-0001"
    assert set(rows[0]) == {"file_id", "file_name", "file_size", "md5",
                            "project", "patient"}
