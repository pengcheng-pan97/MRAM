# MRAM — Multi-Level Recurrent Attention Model (ICONIP 2025)

PyTorch implementation of the ICONIP 2025 paper
**"Emergence of Fixational and Saccadic Movements in a Multi-Level Recurrent
Attention Model for Vision"** by Pengcheng Pan, Shogo Yonekura, and Yasuo Kuniyoshi.

- 📄 Paper (Springer): https://doi.org/10.1007/978-981-95-4378-6_21
- 📝 Preprint (arXiv): https://arxiv.org/abs/2505.13191

This release contains the three hard-attention models compared in the paper:

| Model | Description |
|---|---|
| **RAM**  | Mnih et al. 2014 baseline — single LSTM. |
| **DRAM** | Ba 2015 baseline — two stacked LSTMs + a CNN context init for the upper state. |
| **MRAM** | **Main contribution.** Two stacked LSTMs without gating. Lower layer drives the location policy (SC-like, fast saccadic timescale), upper layer drives classification (visual-cortex-like, slow recognition timescale). The REINFORCE baseline is a hybrid MLP over `concat([h_t1, h_t2])` (eq. 8 of the paper). |

## Install

```bash
pip install -r requirements.txt
```

## Datasets

- **MNIST**, **FashionMNIST**: downloaded automatically by `torchvision` on first run into `--data_dir` (default `./data`).
- **FER2013**: not redistributed here. Place the standard 7-class layout under `data/fer2013/{train,test}/<class>/*.jpg`.

## Train

```bash
# MRAM @ MNIST, 7 glimpses (Table 1, "MRAM, 7 glimpses" — 99.20%)
python main.py --model_choose MRAM --data_choose MNIST \
               --num_glimpses 7 --patch_size 8 --glimpse_scale 1 \
               --hidden_size 256 --batch_size 128 --init_lr 3e-4 \
               --epochs 300 --random_seed 1 \
               --ckpt_dir ./runs/mram_mnist/ckpt \
               --logs_dir ./runs/mram_mnist/logs \
               --plot_dir ./runs/mram_mnist/plots

# MRAM @ FashionMNIST, 12 glimpses (Table 2 — 91.79%)
python main.py --model_choose MRAM --data_choose FashionMNIST \
               --num_glimpses 12 --patch_size 8 --glimpse_scale 1 \
               --hidden_size 256 --epochs 300 \
               --ckpt_dir ./runs/mram_fmnist/ckpt \
               --logs_dir ./runs/mram_fmnist/logs \
               --plot_dir ./runs/mram_fmnist/plots

# MRAM @ FER2013, 12 glimpses, 1 scale (Table 3 — 48.73%)
python main.py --model_choose MRAM --data_choose FER \
               --num_glimpses 12 --patch_size 8 --glimpse_scale 1 \
               --hidden_size 256 --epochs 300 \
               --ckpt_dir ./runs/mram_fer/ckpt \
               --logs_dir ./runs/mram_fer/logs \
               --plot_dir ./runs/mram_fer/plots
```

Swap `--model_choose MRAM` for `RAM` or `DRAM` to reproduce the baselines.

## Test / Evaluate

After training, point the same `--ckpt_dir` and pass `--is_train False`:

```bash
python main.py --is_train False \
               --model_choose MRAM --data_choose MNIST \
               --num_glimpses 7 --patch_size 8 --glimpse_scale 1 \
               --ckpt_dir ./runs/mram_mnist/ckpt \
               --logs_dir ./runs/mram_mnist/logs \
               --plot_dir ./runs/mram_mnist/plots
```

`test()` prints macro-mAP, macro-F1, top-5, and overall accuracy. It also
writes two pickle files into `--plot_dir`:

- `l_0.p`  — one mini-batch (9 samples × num_glimpses locations), for quick visual checks.
- `l_999.p` — all batches × num_glimpses locations, for quantitative scanpath analysis.

## Reproducing the fixation/saccade KDE (Figure 3)

```bash
# Run test on FashionMNIST first to produce l_999.p, then:
python draw_KDE.py            # consumes the l_999.p produced above
python idt_fixations.py       # I-DT fixation extraction used by draw_KDE
```

## Hyperparameters (paper §4)

| Knob | Value |
|---|---|
| Optimizer | Adam, lr `3e-4`, ReduceLROnPlateau on `-valid_acc` |
| Batch size | 128 |
| Glimpse patch | 8×8, single scale (Table 1/2) or 2 scales (Table 3 FER) |
| Hidden size | 256 (LSTMs in all layers) |
| REINFORCE coef α | **0.01** (eq. 10) |
| Reward | 1 if final classification correct, else 0 (paper §3.4) |
| Random seed | 1 |
| Early stopping | 50 epochs without validation improvement |

## File layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── main.py            # entry point
├── config.py          # argparse
├── trainer.py         # training / validation / test loops
├── model.py           # RAM, DRAM, MRAM
├── modules.py         # Retina, GlimpseNetwork, Location/Action/Baseline heads
├── data_loader.py     # MNIST / FashionMNIST / FER
├── utils.py           # AverageMeter, dirs, plotting helper
├── draw_KDE.py        # Figure 3 — fixation-duration + saccade-distance KDE
└── idt_fixations.py   # I-DT fixation segmenter used by draw_KDE.py
```

## Citation

If you use this code or build on the model, please cite the paper:

```bibtex
@inproceedings{pan2026mram,
  author    = {Pan, Pengcheng and Yonekura, Shogo and Kuniyoshi, Yasuo},
  editor    = {Taniguchi, Tadahiro and Leung, Chi Sing Andrew and Kozuno, Tadashi
               and Yoshimoto, Junichiro and Mahmud, Mufti and Doborjeh, Maryam
               and Doya, Kenji},
  title     = {Emergence of Fixational and Saccadic Movements in a Multi-level
               Recurrent Attention Model for Vision},
  booktitle = {Neural Information Processing (ICONIP 2025)},
  series    = {Lecture Notes in Computer Science},
  volume    = {16310},
  pages     = {299--313},
  year      = {2026},
  publisher = {Springer Nature Singapore},
  address   = {Singapore},
  isbn      = {978-981-95-4378-6},
  doi       = {10.1007/978-981-95-4378-6_21}
}
```

## Acknowledgements

This implementation is built on top of
[kevinzakka/recurrent-visual-attention](https://github.com/kevinzakka/recurrent-visual-attention),
Kevin Zakka's PyTorch implementation of the Recurrent Attention Model, released
under the MIT License. The retina / glimpse-network modules, the training
scaffold, and several utilities derive from that project. The DRAM and MRAM
models, the multi-level recurrent architecture, the hybrid REINFORCE baseline,
and the fixation / saccade analysis are contributions of this work.

## License

Released under the MIT License — see [LICENSE](LICENSE). The license retains the
original copyright of Kevin Zakka (2020) for the upstream RAM code and adds the
paper authors' copyright (2025) for the modifications.

> The Springer-published paper PDF is © Springer and is **not** redistributed in
> this repository. Please obtain it via the
> [Springer DOI](https://doi.org/10.1007/978-981-95-4378-6_21) or read the
> [arXiv preprint](https://arxiv.org/abs/2505.13191).
