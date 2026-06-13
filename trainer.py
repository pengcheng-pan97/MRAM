"""ICONIP — Trainer for RAM / DRAM / MRAM.

Hybrid loss (paper eq. 10):
    L = L_classification + L_baseline + alpha * L_REINFORCE
with alpha = 0.01 as reported in §4 of the ICONIP paper.
"""
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import RecurrentAttention, DRAM, MRAM
from utils import AverageMeter


ALPHA_REINFORCE = 0.01  # ICONIP §4: weight of REINFORCE term in the hybrid loss


class Trainer:
    """Trainer for RAM / DRAM / MRAM hard-attention models."""

    def __init__(self, config, data_loader):
        self.config = config
        self.device = torch.device("cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # REINFORCE params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        ds_info = {
            "MNIST":        (10, 1),
            "FashionMNIST": (10, 1),
            "FER":          (7, 3),
        }
        if config.data_choose not in ds_info:
            raise ValueError(
                f"This release supports only MNIST / FashionMNIST / FER, got {config.data_choose!r}."
            )
        self.num_classes, self.num_channels = ds_info[config.data_choose]

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr = config.init_lr
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.plot_dir = config.plot_dir + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "{}_{}_{}x{}_{}".format(
            config.model_name,
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + "/"
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build the model
        model_args = (
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        if config.model_choose == "RAM":
            self.model = RecurrentAttention(*model_args)
        elif config.model_choose == "DRAM":
            self.model = DRAM(*model_args)
        elif config.model_choose == "MRAM":
            self.model = MRAM(*model_args)
        else:
            raise ValueError(
                f"This release supports only RAM / DRAM / MRAM, got {config.model_choose!r}."
            )
        self.model.to(self.device)
        print(f"total parameters: {sum(p.numel() for p in self.model.parameters())}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.init_lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=self.lr_patience)

    # ------------------------------------------------------------------
    #  hidden-state factory
    # ------------------------------------------------------------------
    def reset(self):
        zeros = lambda: torch.zeros(
            1, self.batch_size, self.hidden_size,
            dtype=torch.float, device=self.device, requires_grad=True,
        )
        h_t, h_h, c_t, c_h = zeros(), zeros(), zeros(), zeros()
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True
        return h_t, h_h, c_t, c_h, l_t

    # ------------------------------------------------------------------
    #  training loop
    # ------------------------------------------------------------------
    def train(self):
        if self.resume:
            self.load_checkpoint(best=False)

        print(f"\n[*] Train on {self.num_train} samples, validate on {self.num_valid} samples")
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\nEpoch: {epoch + 1}/{self.epochs} - LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate(epoch)
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
            msg = "train loss: {:.3f} - train acc: {:.3f} - val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg += " [*]"
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc))

            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return

            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def train_one_epoch(self, epoch):
        self.model.train()
        losses, accs, rewards = AverageMeter(), AverageMeter(), AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)

                self.batch_size = x.shape[0]
                h_t, h_h, c_t, c_h, l_t = self.reset()

                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    h_t, h_h, c_t, c_h, l_t, b_t, log_probas, p, *_ = self.model(
                        x, l_t, h_t, h_h, c_t, c_h, self.std
                    )
                    baselines.append(b_t)
                    log_pi.append(p)

                h_t, h_h, c_t, c_h, l_t, b_t, log_probas, p, *_ = self.model(
                    x, l_t, h_t, h_h, c_t, c_h, self.std
                )
                log_pi.append(p)
                baselines.append(b_t)

                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)

                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)

                loss_action = F.nll_loss(log_probas, y)
                loss_baseline = F.mse_loss(baselines, R)
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.mean(torch.sum(-log_pi * adjusted_reward, dim=1))
                loss = loss_action + loss_baseline + ALPHA_REINFORCE * loss_reinforce

                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                losses.update(loss.item(), x.size(0))
                accs.update(acc.item(), x.size(0))
                rewards.update(R.mean().item(), x.size(0))

                loss.backward()
                self.optimizer.step()

                toc = time.time()
                pbar.set_description(
                    "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(toc - tic, loss.item(), acc.item())
                )
                pbar.update(self.batch_size)

                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)
                    log_value("train_reward", rewards.avg, iteration)

        return losses.avg, accs.avg

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        losses, accs = AverageMeter(), AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.repeat(self.M, 1, 1, 1)

            self.batch_size = x.shape[0]
            h_t, h_h, c_t, c_h, l_t = self.reset()

            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                h_t, h_h, c_t, c_h, l_t, b_t, log_probas, p, *_ = self.model(
                    x, l_t, h_t, h_h, c_t, c_h, self.std
                )
                baselines.append(b_t)
                log_pi.append(p)

            h_t, h_h, c_t, c_h, l_t, b_t, log_probas, p, *_ = self.model(
                x, l_t, h_t, h_h, c_t, c_h, self.std
            )
            log_pi.append(p)
            baselines.append(b_t)

            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1]).mean(0)
            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1]).mean(0)
            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1]).mean(0)

            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.mean(torch.sum(-log_pi * adjusted_reward, dim=1))
            loss = loss_action + loss_baseline + ALPHA_REINFORCE * loss_reinforce

            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))
            losses.update(loss.item(), x.size(0))
            accs.update(acc.item(), x.size(0))

            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg

    # ------------------------------------------------------------------
    #  test
    # ------------------------------------------------------------------
    @torch.no_grad()
    def test(self):
        """Evaluate the best checkpoint on the test set.

        Prints: macro-mAP, macro-F1, top-5 accuracy, and overall accuracy.
        Saves the glimpse coordinates of one mini-batch (`l_0.p`) for visual
        inspection, and of all batches (`l_999.p`) for quantitative scanpath
        analysis (see `draw_KDE.py`).
        """
        import pickle

        self.load_checkpoint(best=self.best)
        self.model.eval()

        correct = 0
        all_probs, all_targets = [], []
        locs_quick, locs_all = [], []
        start = time.time()

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.repeat(self.M, 1, 1, 1)

            self.batch_size = x.shape[0]
            h_t, h_h, c_t, c_h, l_t = self.reset()

            for t in range(self.num_glimpses - 1):
                h_t, h_h, c_t, c_h, l_t, b_t, log_probas, p, *_ = self.model(
                    x, l_t, h_t, h_h, c_t, c_h, self.std
                )
                locs_all.append(l_t)
                if i == 0:
                    locs_quick.append(l_t[:9])

            h_t, h_h, c_t, c_h, l_t, b_t, log_probas, p, *_ = self.model(
                x, l_t, h_t, h_h, c_t, c_h, self.std
            )
            locs_all.append(l_t)
            if i == 0:
                locs_quick.append(l_t[:9])

            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1]).mean(0)
            probs = torch.softmax(log_probas, dim=1)
            pred = log_probas.argmax(1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).cpu().sum().item()

            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

        elapsed = time.time() - start
        print(f"infer time per glimpse: {elapsed / (len(self.test_loader) * self.batch_size * self.num_glimpses):.6f}s")

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()
        y_true = np.eye(self.num_classes)[targets]
        mAP = average_precision_score(y_true, probs, average="macro")
        f1 = f1_score(targets, probs.argmax(1), average="macro")
        topk = 5
        top5 = (torch.tensor(probs).topk(topk, dim=1).indices == torch.tensor(targets).view(-1, 1)).any(1).float().mean().item() * 100
        acc = 100.0 * correct / self.num_test
        print(f"[*] Test Acc: {correct}/{self.num_test} ({acc:.2f}% — err {100 - acc:.2f}%)")
        print(f"    mAP: {mAP:.3f}  F1: {f1:.3f}  top-{topk}: {top5:.2f}%")

        # save scanpath outputs for downstream analysis
        with open(self.plot_dir + "l_0.p", "wb") as f:
            pickle.dump([l.cpu().numpy() for l in locs_quick], f)
        with open(self.plot_dir + "l_999.p", "wb") as f:
            pickle.dump([l.cpu().numpy() for l in locs_all], f)

    # ------------------------------------------------------------------
    #  checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, state, is_best):
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, self.model_name + "_model_best.pth.tar"))

    def load_checkpoint(self, best=False):
        print(f"[*] Loading model from {self.ckpt_dir}")
        filename = self.model_name + ("_model_best.pth.tar" if best else "_ckpt.pth.tar")
        ckpt = torch.load(os.path.join(self.ckpt_dir, filename))
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        print(f"[*] Loaded {filename} @ epoch {ckpt['epoch']} (best valid acc {ckpt['best_valid_acc']:.3f})")
