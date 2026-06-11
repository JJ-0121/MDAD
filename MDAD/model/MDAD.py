import torch
import torch.nn as nn
import torch.nn.functional as F

from .RevIN import RevIN
from .decomposition import Decomposition


def topk_softmax(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k >= logits.size(-1):
        return torch.softmax(logits, dim=-1)

    top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
    sparse_logits = torch.full_like(logits, -1e9)
    sparse_logits.scatter_(-1, top_idx, top_vals)
    return torch.softmax(sparse_logits, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Adapter(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualFuse(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([x, residual], dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        update = self.update(fused)
        return x + gate * update


class TokenGate(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        hidden_dim = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([a, b, c], dim=-1)))


class Model(nn.Module):
    def __init__(
        self,
        c_in,
        seq_len,
        d_model,
        dropout,
        wavelet_name,
        wavelet_level,
        num_prototypes,
        proto_top_k,
        n_heads,
        use_revin,
        encoder_layers=1,
        lambda_weight=None,
        causal_loss_weight=None,
        proto_loss_weight=None,
        scale_patch_len=8,
        scale_patch_stride_mode="half",
        device=None,
    ):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        self.proto_top_k = max(1, min(proto_top_k, num_prototypes))
        if lambda_weight is None:
            lambda_weight = 1.0 if causal_loss_weight is None else causal_loss_weight
        self.lambda_weight = float(lambda_weight)
        self.proto_loss_weight = 0.05 if proto_loss_weight is None else float(proto_loss_weight)
        self.scale_patch_len = int(scale_patch_len)
        self.scale_patch_stride_mode = str(scale_patch_stride_mode).lower()
        self.pred_hint_scale = 0.25
        self.tail_buffer_size = 131072
        self.use_revin = bool(use_revin)
        self.device = device if device is not None else torch.device("cpu")
        self._fusion_logged = False

        if self.use_revin:
            self.revin = RevIN(self.c_in)

        self.decomposition = Decomposition(
            input_length=self.seq_len,
            wavelet_name=wavelet_name,
            level=wavelet_level,
            c_in=self.c_in,
            device=self.device,
            no_decomposition=False,
            use_amp=False,
        )
        self.band_lengths = self.decomposition.input_w_dim[1:] + [self.decomposition.input_w_dim[0]]
        self.num_scales = len(self.band_lengths)

        self.scale_patch_cfgs = [self._build_scale_patch_config(length) for length in self.band_lengths]
        self.scale_patch_dims = [self.c_in * cfg["patch_len"] for cfg in self.scale_patch_cfgs]
        self.scale_embed = nn.ModuleList(
            [
                MLP(
                    in_dim=patch_dim,
                    hidden_dim=max(patch_dim // 2, d_model),
                    out_dim=d_model,
                    dropout=dropout,
                )
                for patch_dim in self.scale_patch_dims
            ]
        )
        self.scale_decode = nn.ModuleList(
            [
                MLP(
                    in_dim=d_model,
                    hidden_dim=max(patch_dim // 2, d_model),
                    out_dim=patch_dim,
                    dropout=dropout,
                )
                for patch_dim in self.scale_patch_dims
            ]
        )
        self.scale_pos_embed = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, len(cfg["positions"]), d_model) * 0.02)
                for cfg in self.scale_patch_cfgs
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scale_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(encoder_layers)),
        )

        self.regime_bank = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.05)
        self.koopman_bank = nn.Parameter(torch.empty(num_prototypes, d_model, d_model))
        nn.init.xavier_uniform_(self.koopman_bank)
        self.koopman_bias = nn.Parameter(torch.zeros(num_prototypes, d_model))

        self.scale_to_shared = nn.ModuleList([Adapter(d_model=d_model, dropout=dropout) for _ in range(self.num_scales)])
        self.scale_from_shared = nn.ModuleList([Adapter(d_model=d_model, dropout=dropout) for _ in range(self.num_scales)])
        self.scale_from_pred = nn.ModuleList([Adapter(d_model=d_model, dropout=dropout) for _ in range(self.num_scales)])
        self.scale_pred_gate = nn.ModuleList([TokenGate(d_model=d_model, dropout=dropout) for _ in range(self.num_scales)])
        self.scale_recon_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_scales)])
        self.scale_recon_fuse = nn.ModuleList([ResidualFuse(d_model=d_model, dropout=dropout) for _ in range(self.num_scales)])

        # Kept only for backward-compatible checkpoint loading.
        self.register_buffer("tail_ref_rec", torch.zeros(self.tail_buffer_size))
        self.register_buffer("tail_ref_dyn", torch.zeros(self.tail_buffer_size))
        self.register_buffer("tail_ref_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("tail_ref_count", torch.zeros(1, dtype=torch.long))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._fusion_logged = False
        return result

    def _build_scale_patch_config(self, length: int):
        if length < 2:
            patch_len = length
        else:
            patch_len = min(length, max(2, self.scale_patch_len))

        if self.scale_patch_stride_mode == "full":
            stride = patch_len
        elif self.scale_patch_stride_mode == "half":
            stride = max(1, patch_len // 2)
        else:
            raise ValueError(
                f"Unsupported scale_patch_stride_mode: {self.scale_patch_stride_mode}"
            )

        positions = list(range(0, max(1, length - patch_len + 1), stride))
        last_start = max(0, length - patch_len)
        if positions[-1] != last_start:
            positions.append(last_start)
        return {"patch_len": patch_len, "stride": stride, "positions": positions, "length": length}

    def _extract_patches(self, band: torch.Tensor, cfg):
        patches = [band[:, :, start : start + cfg["patch_len"]] for start in cfg["positions"]]
        patches = torch.stack(patches, dim=1)
        return patches.reshape(band.size(0), len(cfg["positions"]), self.c_in * cfg["patch_len"])

    def _reconstruct_band(self, patch_hat: torch.Tensor, cfg):
        patch_hat = patch_hat.view(
            patch_hat.size(0),
            len(cfg["positions"]),
            self.c_in,
            cfg["patch_len"],
        )
        out = patch_hat.new_zeros(patch_hat.size(0), self.c_in, cfg["length"])
        counts = patch_hat.new_zeros(patch_hat.size(0), self.c_in, cfg["length"])

        for idx, start in enumerate(cfg["positions"]):
            out[:, :, start : start + cfg["patch_len"]] += patch_hat[:, idx]
            counts[:, :, start : start + cfg["patch_len"]] += 1.0

        return out / counts.clamp_min(1.0)

    def _values_to_time(self, values: torch.Tensor, target_len: int):
        if values.size(1) == target_len:
            return values
        return F.interpolate(values.unsqueeze(1), size=target_len, mode="linear", align_corners=False).squeeze(1)

    def _stable_koopman_bank(self):
        flat_norm = self.koopman_bank.flatten(1).norm(dim=-1).clamp_min(1.0)
        return 0.98 * self.koopman_bank / flat_norm.unsqueeze(-1).unsqueeze(-1)

    def _koopman_rollout(self, shared_tokens, regime_tokens, weights, stable_bank):
        batch_size = shared_tokens.size(0)
        num_steps = shared_tokens.size(1)
        if num_steps < 2:
            zero_map = shared_tokens.new_zeros(batch_size, self.seq_len)
            return regime_tokens, zero_map, shared_tokens.new_tensor(0.0)

        # One-step ablation: keep only t -> t+1 Koopman prediction.
        current_tokens = regime_tokens[:, :-1]
        step_weights = weights[:, :-1]
        operator = torch.einsum("btk,kde->btde", step_weights, stable_bank)
        bias = torch.einsum("btk,kd->btd", step_weights, self.koopman_bias)
        one_step_pred = torch.einsum("btd,btde->bte", current_tokens, operator) + bias

        target = shared_tokens[:, 1:]
        dyn_measure = (one_step_pred - target).abs().mean(dim=-1)
        dyn_map = self._values_to_time(torch.cat([dyn_measure[:, :1], dyn_measure], dim=1), self.seq_len)

        pred_tokens = torch.cat([regime_tokens[:, :1], one_step_pred], dim=1)

        return pred_tokens, dyn_map, dyn_measure.mean()

    def _encode_scale_tokens(self, bands_target):
        scale_tokens = []
        for idx, band in enumerate(bands_target):
            patches = self._extract_patches(band, self.scale_patch_cfgs[idx])
            tokens = self.scale_embed[idx](patches)
            tokens = tokens + self.scale_pos_embed[idx]
            tokens = self.scale_encoder(tokens)
            scale_tokens.append(tokens)
        return scale_tokens

    def _fuse_scores(self, rec_score: torch.Tensor, dyn_map: torch.Tensor):
        if (not self.training) and (not self._fusion_logged):
            print("Score fusion : multiplicative coupling(rec, dyn)")
            self._fusion_logged = True
        total_score = rec_score * dyn_map
        return total_score

    def forward(self, x):
        x_input = x
        if self.use_revin:
            if self.revin.affine:
                self.revin.affine_weight = self.revin.affine_weight.to(x.device)
                self.revin.affine_bias = self.revin.affine_bias.to(x.device)
            x_norm = self.revin(x, "norm")
        else:
            x_norm = x

        x_norm_bcl = x_norm.transpose(1, 2)
        yl, yh = self.decomposition.transform(x_norm_bcl)
        bands_target = list(yh) + [yl]
        scale_tokens = self._encode_scale_tokens(bands_target)

        stable_bank = self._stable_koopman_bank()
        regime_bank_norm = F.normalize(self.regime_bank, dim=-1, eps=1e-6)

        dyn_losses = []
        dyn_maps = []
        recon_scale_tokens = []

        for scale_idx, tokens in enumerate(scale_tokens):
            shared_tokens = self.scale_to_shared[scale_idx](tokens)
            shared_tokens_norm = F.normalize(shared_tokens, dim=-1, eps=1e-6)
            logits = torch.matmul(shared_tokens_norm, regime_bank_norm.transpose(0, 1))
            weights = topk_softmax(logits, self.proto_top_k)

            regime_shared = torch.matmul(weights, self.regime_bank)
            pred_shared, dyn_map, dyn_loss = self._koopman_rollout(shared_tokens, regime_shared, weights, stable_bank)

            regime_scale = self.scale_from_shared[scale_idx](regime_shared)
            pred_scale = self.scale_from_pred[scale_idx](pred_shared)
            pred_gate = self.scale_pred_gate[scale_idx](tokens, regime_scale, pred_scale)
            recon_hint = regime_scale + self.pred_hint_scale * pred_gate * (pred_scale - regime_scale)

            recon_scale = self.scale_recon_fuse[scale_idx](tokens, recon_hint)
            recon_scale = self.scale_recon_norm[scale_idx](recon_scale)

            dyn_losses.append(dyn_loss)
            dyn_maps.append(dyn_map)
            recon_scale_tokens.append(recon_scale)

        dyn_loss = torch.stack(dyn_losses).mean()
        dyn_map = torch.stack(dyn_maps, dim=0).max(dim=0).values

        bands_hat = []
        for scale_idx, recon_scale in enumerate(recon_scale_tokens):
            patch_hat = self.scale_decode[scale_idx](recon_scale)
            band_hat = self._reconstruct_band(patch_hat, self.scale_patch_cfgs[scale_idx])
            bands_hat.append(band_hat)

        yl_hat = bands_hat[-1]
        yh_hat = bands_hat[:-1]
        x_hat_norm = self.decomposition.inv_transform(yl_hat, yh_hat).transpose(1, 2)

        if self.use_revin:
            x_hat = self.revin(x_hat_norm, "denorm")
        else:
            x_hat = x_hat_norm

        rec_measure = (x_hat - x_input).abs()
        rec_score = rec_measure.mean(dim=-1)
        rec_loss = rec_measure.mean()

        total_score = self._fuse_scores(rec_score, dyn_map)

        if self.lambda_weight > 0:
            total_loss = dyn_loss + rec_loss / self.lambda_weight
        else:
            total_loss = rec_loss

        return {
            "x_hat": x_hat,
            "bands_hat": bands_hat,
            "bands_target": bands_target,
            "losses": {
                "total": total_loss,
                "rec": rec_loss,
                "dyn": dyn_loss,
            },
            "scores": {
                "total": total_score,
                "rec": rec_score,
                "dyn": dyn_map,
            },
        }
