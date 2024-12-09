# Example for preprocessing

## TissueNet Dataset
Raw data can be downloaded from [TissueNet](https://datasets.deepcell.org/). 

Change the path of the raw data and output folder in the `prep_tissuenet.py` script. 

```bash
python prep_tissuenet.py
```

## CODEX CRC Dataset
Raw data can be downloaded from [CODEX CRC](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70227790).

Annotation data can be downloaded from [CODEX CRC Annotation](https://data.mendeley.com/datasets/mpjzbtfgfr/1)

Change the corresponding paths in the `prep_codex.py` script. 

```bash
python prep_codex.py
```

## Xenium Dataset
Example raw data can be downloaded from [Xenium](https://www.10xgenomics.com/datasets?query=Xenium&page=1&configure%5BhitsPerPage%5D=50&configure%5BmaxValuesPerFacet%5D=1000).

Used the following  scripts to preprocess the data. 

1. `prep_xenium_fov.py`
2. `prep_xenium_patch.py`
3. `prep_xenium_input.py`
