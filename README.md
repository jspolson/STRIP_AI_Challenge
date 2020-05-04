# PANDA_Challenge
A repository for collaborating on PANDA challenge.
### docker: Contains Dockerfile for code development.
* Solved tqdm and jupyter notebook dependency issue by（already added to dockerfile）
```bash
conda install -c conda-forge ipywidgets
pip install --upgrade jupyter_client
```

* Solved skimage open tiff image issue by installing latest tiff package (already added to dockerfile)
```bash
pip install tifffile
```
### data：holds data files
### preprocessing:


