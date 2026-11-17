# Spiking optical flow on DSEC (INT8)

Fully spiking SiLIF network (SpikingJelly), 973k parameters, INT8 weights.
Val EPE 1.49 px on the 5 DSEC val recordings.

## 1. Dependencies

```bash
conda env create -f environment.yml
conda activate qflownet
```

Runs on GPU or CPU.

## 2. Data

Download `train_events.zip` and `train_optical_flow.zip` from https://dsec.ifi.uzh.ch/dsec-datasets/download/ and unzip both in the same folder `<DSEC>`.

Set `DATA_ROOT = Path("<DSEC>")` in `code/dsec_optical/flow_cache.py`.

Build the cache from `code/` (CPU, ~10 min, ~18 GB in `<DSEC>/flow_cache/`):

```bash
python -m dsec_optical.flow_cache --bins 15 --downsample 2 --workers 18
```

Check:

```bash
python -c "
from dsec_optical.flow_cache import CachedFlowDataset as D, FLOW_TRAIN_RECORDINGS as T, _FLOW_VAL_RECORDINGS as V
print(len(D(T)), len(D(V)))"
```

should print `6013 2157`.

## 3. Inference

From `code/`:

```python
import torch
from types import SimpleNamespace
from train_fast_flow import apply_qat, build_model
from dsec_optical.flow_cache import CachedFlowDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("../checkpoint/h200_p3_int8_D_j33273.pt", map_location=dev, weights_only=False)
a = SimpleNamespace(**ck["args"])
model = build_model(a)
apply_qat(model, a.qat_bits, a.qat_act_bits)
model.load_state_dict(ck["model"])
model.to(dev).eval()

ds = CachedFlowDataset(["zurich_city_08_a"])
vox, gt, valid, _ = ds[0]                 # vox (15, 240, 320)
with torch.no_grad():
    flow = model(vox[None].to(dev))[0]    # (1, 2, 480, 640), pixels

epe = (flow[0].cpu() - gt).norm(dim=0)[valid[0]].mean()
print(epe)
```

Figure:

```bash
python plot_fast_flow.py --ckpt ../checkpoint/h200_p3_int8_D_j33273.pt
```

Video on one val recording (+ EPE and latency):

```bash
python visualize_dsec_flow_video.py --fast_ckpt ../checkpoint/h200_p3_int8_D_j33273.pt \
    --val_seq zurich_city_08_a --out_mp4 out.mp4 --out_gif out.gif
```

Val recordings: thun_00_a, zurich_city_02_d, zurich_city_03_a, zurich_city_08_a, zurich_city_11_b.

