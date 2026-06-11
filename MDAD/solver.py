import importlib
import inspect
import math
import os
import time
import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data_factory.data_loader import get_loader_segment
from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events


def adjust_learning_rate(optimizer, epoch, lr_, lradj="type1", train_epochs=None, verbose=True):
    if lradj == "type1":
        lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "type3":
        lr_adjust = {
            epoch: lr_ if epoch < 3 else lr_ * (0.9 ** ((epoch - 3) // 1))
        }
    elif lradj == "cosine":
        total_epochs = max(1, int(train_epochs if train_epochs is not None else epoch))
        lr_adjust = {
            epoch: lr_ / 2.0 * (1.0 + math.cos(epoch / total_epochs * math.pi))
        }
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if verbose:
            print("Updating learning rate to {}".format(lr))


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="train",
            dataset=self.dataset,
        )
        self.vali_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="val",
            dataset=self.dataset,
        )
        self.thre_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="thre",
            dataset=self.dataset,
        )

        self.device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"
        )

        self.build_model()

    def build_model(self):
        model_name = getattr(self, "model_name", "wavelet_causal_expert")
        model_module = importlib.import_module(f"model.{model_name}")
        model_cls = getattr(model_module, "Model", None)
        if model_cls is None:
            model_cls = getattr(model_module, "BasicCrossAD")
        model_kwargs = {
            "c_in": self.input_c,
            "seq_len": self.win_size,
            "d_model": self.d_model,
            "dropout": self.dr,
            "wavelet_name": self.wavelet_name,
            "wavelet_level": self.wavelet_level,
            "num_prototypes": self.num_prototypes,
            "proto_top_k": self.proto_top_k,
            "n_heads": self.n_heads,
            "use_revin": getattr(self, "use_revin", None),
            "band_aux_weight": getattr(self, "band_aux_weight", None),
            "proto_loss_weight": getattr(self, "proto_loss_weight", None),
            "encoder_layers": getattr(self, "encoder_layers", None),
            "lambda_weight": getattr(self, "lambda_weight", None),
            "causal_loss_weight": getattr(self, "causal_loss_weight", None),
            "scale_patch_len": getattr(self, "scale_patch_len", None),
            "scale_patch_stride_mode": getattr(self, "scale_patch_stride_mode", None),
            "device": self.device,
        }
        signature = inspect.signature(model_cls.__init__)
        filtered_kwargs = {
            key: value
            for key, value in model_kwargs.items()
            if key in signature.parameters
        }
        self.model = model_cls(**filtered_kwargs).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )

    def _checkpoint_path(self):
        return os.path.join(
            str(self.model_save_path),
            str(self.dataset) + "_checkpoint.pth",
        )

    def _energy_from_output(self, model_out, input_data):
        if isinstance(model_out, dict):
            scores = model_out.get("scores")
            if scores is not None:
                if "total" in scores:
                    return scores["total"]
                if "rec" in scores:
                    return scores["rec"]

            x_hat = model_out.get("x_hat")
            if x_hat is not None:
                return (x_hat - input_data).abs().mean(dim=-1)

        return (model_out - input_data).pow(2).mean(dim=-1)

    def _score_branches_from_output(self, model_out, input_data):
        if isinstance(model_out, dict):
            scores = model_out.get("scores")
            if scores is not None:
                return {
                    key: value
                    for key, value in scores.items()
                    if torch.is_tensor(value)
                }

            x_hat = model_out.get("x_hat")
            if x_hat is not None:
                rec = (x_hat - input_data).abs().mean(dim=-1)
                return {"rec": rec, "total": rec}

        rec = (model_out - input_data).pow(2).mean(dim=-1)
        return {"rec": rec, "total": rec}

    def _collect_energy(self, loader, with_labels=False):
        energies = []
        labels_all = []

        with torch.no_grad():
            for input_data, labels in loader:
                input_data = input_data.float().to(self.device)
                model_out = self.model(input_data)
                energy = self._energy_from_output(model_out, input_data)
                energies.append(energy.detach().cpu().numpy())

                if with_labels:
                    if torch.is_tensor(labels):
                        labels = labels.detach().cpu().numpy()
                    labels_all.append(labels)

        energies = np.concatenate(energies, axis=0).reshape(-1)
        if with_labels:
            return energies, np.concatenate(labels_all, axis=0).reshape(-1)
        return energies

    def _collect_branch_scores(self, loader, branch_names, with_labels=False):
        scores_all = {name: [] for name in branch_names}
        labels_all = []

        with torch.no_grad():
            for input_data, labels in loader:
                input_data = input_data.float().to(self.device)
                model_out = self.model(input_data)
                score_dict = self._score_branches_from_output(model_out, input_data)

                for name in branch_names:
                    if name not in score_dict:
                        raise KeyError(f"Required score branch '{name}' not found in model output.")
                    scores_all[name].append(score_dict[name].detach().cpu().numpy())

                if with_labels:
                    if torch.is_tensor(labels):
                        labels = labels.detach().cpu().numpy()
                    labels_all.append(labels)

        scores_all = {
            name: np.concatenate(values, axis=0).reshape(-1)
            for name, values in scores_all.items()
        }
        if with_labels:
            return scores_all, np.concatenate(labels_all, axis=0).reshape(-1)
        return scores_all

    def _affiliation_metrics(self, labels, pred):
        events_pred = convert_vector_to_events(pred)
        events_label = convert_vector_to_events(labels)
        result = pr_from_events(events_pred, events_label, (0, len(pred)))
        precision = result["precision"]
        recall = result["recall"]
        f_score = 2 * precision * recall / (precision + recall + 1e-12)
        return precision, recall, f_score

    def _adjust_predictions(self, pred, labels):
        anomaly_state = False
        for i in range(len(labels)):
            if labels[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if labels[j] == 0:
                        break
                    pred[j] = 1
                for j in range(i, len(labels)):
                    if labels[j] == 0:
                        break
                    pred[j] = 1
            elif labels[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        return pred

    def _best_raw_f1(self, scores, labels):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)

        total_positive = int(labels.sum())
        if scores.size == 0 or total_positive == 0:
            return 0.0, 0.0, 0.0, float("nan")

        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

        if positive_scores.size > 0:
            positive_values, positive_counts = np.unique(positive_scores, return_counts=True)
            positive_gain = {
                float(value): int(count)
                for value, count in zip(positive_values.tolist(), positive_counts.tolist())
            }
        else:
            positive_gain = {}

        if negative_scores.size > 0:
            negative_values, negative_counts = np.unique(negative_scores, return_counts=True)
            negative_gain = {
                float(value): int(count)
                for value, count in zip(negative_values.tolist(), negative_counts.tolist())
            }
        else:
            negative_gain = {}

        candidate_scores = sorted(set(positive_gain.keys()) | set(negative_gain.keys()), reverse=True)
        if not candidate_scores:
            return 0.0, 0.0, 0.0, float("nan")

        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = float("nan")
        tp = 0
        fp = 0

        for score in candidate_scores:
            tp += positive_gain.get(score, 0)
            fp += negative_gain.get(score, 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / total_positive
            f1 = 2 * precision * recall / (precision + recall + 1e-12)

            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = float(np.nextafter(np.float64(score), -np.inf))

        return best_precision, best_recall, best_f1, best_threshold

    def vali(self, vali_loader):
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for input_data, _ in vali_loader:
                input_data = input_data.float().to(self.device)
                model_out = self.model(input_data)
                loss = model_out["losses"]["total"]
                loss_list.append(loss.item())

        return np.average(loss_list)

    def train(self):
        print("======================TRAIN MODE======================")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

        path = self.model_save_path
        os.makedirs(path, exist_ok=True)

        train_steps = len(self.train_loader)
        best_vali_loss = np.inf
        total_train_time = 0.0
        lradj = getattr(self, "lradj", "type1")

        for epoch in range(self.num_epochs):
            start_epoch_time = time.time()
            train_loss_list = []

            self.model.train()

            with tqdm(total=train_steps) as pbar:
                for input_data, _ in self.train_loader:
                    self.optimizer.zero_grad()

                    input_data = input_data.float().to(self.device)
                    model_out = self.model(input_data)
                    loss_terms = model_out["losses"]
                    loss = loss_terms["total"]

                    train_loss_list.append(loss.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                    postfix = {"total": f"{loss.item():.4f}"}
                    for key, value in loss_terms.items():
                        if key != "total" and torch.is_tensor(value):
                            postfix[key] = f"{value.item():.4f}"
                    pbar.set_postfix(**postfix)
                    pbar.update(1)

            train_loss = np.average(train_loss_list)
            vali_loss = self.vali(self.vali_loader)
            epoch_time = time.time() - start_epoch_time
            total_train_time += epoch_time

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss: {3:.7f} | Time: {4:.2f}s".format(
                    epoch + 1, train_steps, train_loss, vali_loss, epoch_time
                )
            )
            print("Average Training Time per Epoch: {:.3f}s".format(total_train_time / (epoch + 1)))

            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                os.makedirs(path, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    self._checkpoint_path(),
                )
                print("Validation loss improved. Saving model ...")

            adjust_learning_rate(
                self.optimizer,
                epoch + 1,
                self.lr,
                lradj=lradj,
                train_epochs=self.num_epochs,
            )

    def test(self):
        ckpt_path = self._checkpoint_path()

        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        print("======================TEST MODE======================")
        train_energy = self._collect_energy(self.train_loader)
        threshold_energy, test_labels = self._collect_energy(self.thre_loader, with_labels=True)
        test_energy = threshold_energy
        combined_energy = np.concatenate([train_energy, threshold_energy], axis=0)

        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        start_test_time = time.time()
        test_labels = np.array(test_labels).astype(int)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        print("Testing Time: {:.3f}s".format(time.time() - start_test_time))

        aff_precision, aff_recall, aff_f1 = self._affiliation_metrics(gt.copy(), pred.copy())
        adjusted_pred = self._adjust_predictions(pred.copy(), gt.copy())
        best_precision, best_recall, best_f1, best_threshold = self._best_raw_f1(
            test_energy,
            gt,
        )

        print("pred: ", adjusted_pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, adjusted_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt,
            adjusted_pred,
            average="binary",
            zero_division=0,
        )
        auc_roc = roc_auc_score(gt, test_energy)
        auc_pr = average_precision_score(gt, test_energy)

        print(
            "AAPrecision : {:0.3f}, AARecall : {:0.3f}, AAF-score : {:0.3f}".format(
                aff_precision,
                aff_recall,
                aff_f1,
            )
        )
        print(
            "Accuracy : {:0.3f}, Precision : {:0.3f}, Recall : {:0.3f}, F-score : {:0.3f}, "
            "AUC-ROC : {:0.4f} , AUC-PR : {:0.4f}".format(
                accuracy,
                precision,
                recall,
                f_score,
                auc_roc,
                auc_pr,
            )
        )
        print(
            "Raw-Best-Precision : {:0.3f}, Raw-Best-Recall : {:0.3f}, Raw-Best-F-score : {:0.3f}, "
            "Raw-Best-Threshold : {:0.8f}".format(
                best_precision,
                best_recall,
                best_f1,
                best_threshold,
            )
        )
        from utils.metrics import metricor
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt(gt, test_energy, 100,250)
        print(f"VUS-ROC: {avg_auc_3d:.4f}, VUS-PR: {avg_ap_3d:.4f}")
        return accuracy, precision, recall, f_score
