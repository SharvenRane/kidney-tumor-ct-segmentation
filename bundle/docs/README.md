# MONAI Bundle: Kidney and Renal Mass CT Segmentation

A schema validated, version pinned MONAI Bundle for the kidney and mass
segmentation model. The bundle makes the model self describing and runnable
without the training scripts.

## Contents

```
configs/metadata.json    version triple, task, and the input/output tensor contract
configs/inference.json    network, preprocessing, sliding window inferer, postprocessing
models/model.pt           trained weights (added after training; not committed to git)
```

## Use

```
# validate the metadata against the MONAI bundle schema
python -m monai.bundle verify_metadata --meta_file configs/metadata.json --filepath eval/schema.json

# confirm the network matches its declared input/output contract
python -m monai.bundle verify_net_in_out network_def --meta_file configs/metadata.json --config_file configs/inference.json

# run inference on a folder of NIfTI cases
python -m monai.bundle run --config_file configs/inference.json --dataset_dir /path/to/cases
```

The model expects single channel contrast CT, oriented RAS, resampled to 1.5 mm
isotropic, intensity windowed to [-200, 300] HU and scaled to [0, 1]. It outputs
a three class labelmap: 0 background, 1 kidney, 2 mass.

Data source: C4KC-KiTS (TCIA), CC BY 3.0,
https://doi.org/10.7937/TCIA.2019.IX49E8NX. Illustrative portfolio artifact, not
a medical device.
