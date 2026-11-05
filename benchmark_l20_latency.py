#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn

INPUT_DIM = 23
SEQ_LEN = 82
HIDDEN_N = 80
THRESHOLD = 56
N_COUNTS = 81
MC_SAMPLES = 64


class SequenceTransformerCount(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, max_len=SEQ_LEN, n_counts=N_COUNTS, hidden_dim=72, num_heads=4, depth=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_counts),
        )

    def forward(self, x):
        h = self.embed(x)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_emb[:, : h.shape[1], :]
        h = self.encoder(h)
        return self.head(h[:, 0, :])


class AttentiveNPTail(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_n=HIDDEN_N, threshold=THRESHOLD, latent_dim=8, hidden_dim=96, num_heads=4):
        super().__init__()
        self.hidden_n = hidden_n
        self.threshold = threshold
        self.latent_dim = latent_dim
        self.context_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.target_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.posterior = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * latent_dim))
        self.item_head = nn.Sequential(nn.LayerNorm(2 * hidden_dim + latent_dim), nn.Linear(2 * hidden_dim + latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x, mc_samples=MC_SAMPLES):
        context = x[:, :-self.hidden_n, :]
        target = x[:, -self.hidden_n :, :]
        ctx = self.context_embed(context)
        tgt = self.target_embed(target)
        attended, _ = self.attn(tgt, ctx, ctx, need_weights=False)
        mu, logvar = self.posterior(ctx.mean(dim=1)).chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -5.0, 2.0)
        batch = x.shape[0]
        eps = torch.randn(batch, mc_samples, self.latent_dim, device=x.device, dtype=x.dtype)
        z = mu[:, None, :] + torch.exp(0.5 * logvar)[:, None, :] * eps
        z = z[:, :, None, :].expand(-1, -1, self.hidden_n, -1)
        item = torch.cat([
            tgt[:, None, :, :].expand(-1, mc_samples, -1, -1),
            attended[:, None, :, :].expand(-1, mc_samples, -1, -1),
            z,
        ], dim=-1)
        probs = torch.sigmoid(self.item_head(item).squeeze(-1))
        return differentiable_tail(probs, self.threshold).mean(dim=-1)


class ConditionalCoupling(nn.Module):
    def __init__(self, latent_dim, cond_dim, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(latent_dim + cond_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2 * latent_dim))

    def forward(self, z, cond):
        masked = z * self.mask
        shift, log_scale = self.net(torch.cat([masked, cond], dim=-1)).chunk(2, dim=-1)
        log_scale = 0.6 * torch.tanh(log_scale)
        return masked + (1.0 - self.mask) * (z * torch.exp(log_scale) + shift)


class FlowLatentTail(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_n=HIDDEN_N, threshold=THRESHOLD, latent_dim=8, hidden_dim=96, num_heads=4):
        super().__init__()
        self.hidden_n = hidden_n
        self.threshold = threshold
        self.latent_dim = latent_dim
        self.context_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.target_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cond_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        masks = [[float((j + layer_idx) % 2) for j in range(latent_dim)] for layer_idx in range(4)]
        self.couplings = nn.ModuleList([ConditionalCoupling(latent_dim, hidden_dim, mask) for mask in masks])
        self.item_head = nn.Sequential(nn.LayerNorm(2 * hidden_dim + latent_dim), nn.Linear(2 * hidden_dim + latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x, mc_samples=MC_SAMPLES):
        context = x[:, :-self.hidden_n, :]
        target = x[:, -self.hidden_n :, :]
        ctx = self.context_embed(context)
        tgt = self.target_embed(target)
        attended, _ = self.attn(tgt, ctx, ctx, need_weights=False)
        cond = self.cond_head(ctx.mean(dim=1))
        batch = x.shape[0]
        z = torch.randn(batch, mc_samples, self.latent_dim, device=x.device, dtype=x.dtype)
        cond_s = cond[:, None, :].expand(-1, mc_samples, -1).reshape(batch * mc_samples, -1)
        z_flat = z.reshape(batch * mc_samples, -1)
        for coupling in self.couplings:
            z_flat = coupling(z_flat, cond_s)
        z = z_flat.reshape(batch, mc_samples, self.latent_dim)
        z = z[:, :, None, :].expand(-1, -1, self.hidden_n, -1)
        item = torch.cat([
            tgt[:, None, :, :].expand(-1, mc_samples, -1, -1),
            attended[:, None, :, :].expand(-1, mc_samples, -1, -1),
            z,
        ], dim=-1)
        probs = torch.sigmoid(self.item_head(item).squeeze(-1))
        return differentiable_tail(probs, self.threshold).mean(dim=-1)


def differentiable_tail(probs, threshold):
    batch, components, n_items = probs.shape
    dp = probs.new_zeros(batch, components, n_items + 1)
    dp[:, :, 0] = 1.0
    for idx in range(n_items):
        p = probs[:, :, idx]
        new = probs.new_zeros(batch, components, n_items + 1)
        new[:, :, 1:] += dp[:, :, :-1] * p[:, :, None]
        new[:, :, :-1] += dp[:, :, :-1] * (1.0 - p[:, :, None])
        dp = new
    return dp[:, :, threshold:].sum(dim=-1)


def factories():
    return {
        "SeqTXCount": SequenceTransformerCount,
        "ANP": AttentiveNPTail,
        "FlowLatent": FlowLatentTail,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def decision_forward(name, model, x):
    if name == "SeqTXCount":
        logits = model(x)
        return torch.softmax(logits, dim=-1)[:, THRESHOLD:].sum(dim=-1)
    return model(x, mc_samples=MC_SAMPLES)


def percentile(values, q):
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def benchmark_one(name, batch_size, warmup, repeats, trials, device):
    all_ms = []
    trial_rows = []
    peak_mib = 0.0
    params = None
    for trial in range(trials):
        torch.manual_seed(34090 + trial)
        torch.cuda.manual_seed_all(34090 + trial)
        model = factories()[name]().to(device).eval()
        params = count_parameters(model)
        x = torch.randn(batch_size, SEQ_LEN, INPUT_DIM, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                decision_forward(name, model, x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                y = decision_forward(name, model, x)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            torch.cuda.synchronize()
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / (1024.0 ** 2))
            if not bool(torch.isfinite(y).all()):
                raise RuntimeError(f"non-finite output for {name}")
        all_ms.extend(samples)
        trial_rows.append({
            "trial": trial,
            "mean_batch_ms": statistics.fmean(samples),
            "median_batch_ms": statistics.median(samples),
            "p95_batch_ms": percentile(samples, 0.95),
            "mean_ms_per_slate": statistics.fmean(samples) / batch_size,
        })
        del model, x, y
        torch.cuda.empty_cache()
        time.sleep(0.2)
    trial_means = [row["mean_ms_per_slate"] for row in trial_rows]
    return {
        "parameters": params,
        "batch_size": batch_size,
        "warmup_per_trial": warmup,
        "repeats_per_trial": repeats,
        "trials": trials,
        "total_timed_batches": len(all_ms),
        "mean_batch_ms": statistics.fmean(all_ms),
        "sd_batch_ms": statistics.stdev(all_ms) if len(all_ms) > 1 else 0.0,
        "median_batch_ms": statistics.median(all_ms),
        "p95_batch_ms": percentile(all_ms, 0.95),
        "mean_ms_per_slate": statistics.fmean(all_ms) / batch_size,
        "median_ms_per_slate": statistics.median(all_ms) / batch_size,
        "p95_ms_per_slate": percentile(all_ms, 0.95) / batch_size,
        "between_trial_sd_ms_per_slate": statistics.stdev(trial_means) if len(trial_means) > 1 else 0.0,
        "peak_allocated_mib": peak_mib,
        "trial_summaries": trial_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="l20_latency.json")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup-b1", type=int, default=50)
    parser.add_argument("--repeats-b1", type=int, default=200)
    parser.add_argument("--warmup-b192", type=int, default=20)
    parser.add_argument("--repeats-b192", type=int, default=50)
    args = parser.parse_args()
    checks = {name: count_parameters(factory()) for name, factory in factories().items()}
    expected = {"SeqTXCount": 145377, "ANP": 91329, "FlowLatent": 227153}
    if checks != expected:
        raise RuntimeError(f"parameter check failed: {checks} != {expected}")
    if args.check:
        print(json.dumps({"parameter_check": checks, "shape": [SEQ_LEN, INPUT_DIM], "hidden_n": HIDDEN_N, "threshold": THRESHOLD, "mc_samples": MC_SAMPLES}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(device)
    result = {
        "protocol": {
            "timing": "CUDA events with end-event synchronization; inputs preloaded on GPU",
            "scope": "head-only decision output; excludes data loading, host-device transfer, and TaxScore construction",
            "input_shape_per_slate": [SEQ_LEN, INPUT_DIM],
            "ml1m_setting": {"k": 2, "n": 80, "c": 0.7, "threshold": THRESHOLD},
            "posterior_samples": MC_SAMPLES,
            "precision": "float32",
            "tf32": True,
        },
        "environment": {
            "gpu": props.name,
            "gpu_memory_mib": props.total_memory / (1024.0 ** 2),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "models": {},
    }
    plans = [(1, args.warmup_b1, args.repeats_b1), (192, args.warmup_b192, args.repeats_b192)]
    for name in factories():
        result["models"][name] = {}
        for batch_size, warmup, repeats in plans:
            print(f"benchmarking {name} batch={batch_size}", flush=True)
            result["models"][name][f"batch_{batch_size}"] = benchmark_one(name, batch_size, warmup, repeats, args.trials, device)
    payload = json.dumps(result, indent=2)
    Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
