# FlowCast-ODE

**License:** [MIT](LICENSE)

This repository contains the **source code** for the paper:

**FlowCast-ODE: Continuous hourly weather forecasting with dynamic flow matching**

**Authors**  
Shuangshuang He, Yuanting Zhang*, Shuo Wang, Hongli Liang, Qingye Meng, Xingyuan Yuan, and Jingfang Fan*

The asterisk (\*) indicates corresponding authors.

## Model weights & preprocessing stats (Hugging Face)

**Model weights** and **preprocessed statistics** are available on Hugging Face: [Caiyun-Weather/FlowCast-ODE](https://huggingface.co/Caiyun-Weather/FlowCast-ODE).

## Repository structure

```
.
├── configs/
│   ├── experiment/     # f6_expflow, f6_expflow_predict, f6_exppangu, …
│   ├── data/           # e.g. era5Global*.yaml (global stack)
│   ├── model/          # e.g. flowmatching6hr.yaml, pangu.yaml
│   ├── train.yaml      # default datamodule + Lightning entry (see keys above)
│   ├── eval.yaml
│   └── predict.yaml
├── data/               # user data (see data/README.md)
├── prepare/            # optional ERA5 prep (e.g. retrieve_era5_1degree.py)
├── src/                # train / eval / predict, models, data, utils
├── tests/              # hydra config smoke tests (no real data)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Requirements

- **Python** 3.10+ (3.8–3.10 also common with the upstream Lightning–Hydra template)
- **PyTorch** (install a build that matches your CUDA/CPU; see [pytorch.org](https://pytorch.org/))
- **Python packages:** `pip install -r requirements.txt`


## Quick start (training)

From the **repository root** (directory containing `src/` and `configs/`):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

1. **Prepare** data under `data/` and set paths in `configs/data/*.yaml` and the dataset block of your experiment to match your layout (including fine-tune vs pretrain splits if you replicate the paper’s two-stage schedule).

2. **Flow-matching** (same stack as `f6_expflow`; default entries in `train.yaml` use the sharded global datamodule from the repo):

   ```bash
   python src/train.py experiment=f6_expflow trainer=gpu
   ```

3. **Pangu baseline** (shared global datamodule class; config variants in the table above):

   ```bash
   python src/train.py experiment=f6_exppangu trainer=gpu
   ```

4. **Prediction** with a trained checkpoint (set path explicitly; the bundled `f6_expflow_predict` config has `ckpt_path: null` for portability):

   ```bash
   python src/predict.py experiment=f6_expflow_predict trainer=gpu ckpt_path=/path/to/epoch_XXX.ckpt
   ```

Outputs follow Hydra’s `outputs/` layout (see `configs/hydra/default.yaml`). Checkpoints and any results go under each run’s `${paths.output_dir}` as configured in the experiment.


## Citation
 You can use this **placeholder** BibTeX (fill in `journal` and other fields when the paper is published):

```bibtex
@article{flowcastode2026,
  title={FlowCast-ODE: Continuous hourly weather forecasting with dynamic flow matching},
  author={He, Shuangshuang and Zhang, Yuanting e.a.},
  journal={},
  year={2026}
}
```

## Contact

For questions about the code, open an **Issue** on GitHub or email [heshuangshuang816@gmail.com](mailto:heshuangshuang816@gmail.com). 

## Acknowledgments

The training layout builds on the [Lightning-Hydra](https://github.com/ashleve/lightning-hydra-template) style template (boilerplate and Hydra structure).
