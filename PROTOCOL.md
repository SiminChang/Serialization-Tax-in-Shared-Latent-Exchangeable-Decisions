# MovieLens TaxScore Audit Protocol

This document describes the implemented MovieLens slate tail-risk audit.
Candidate construction and grid-based routing are separate operations. Each uses
training information and revealed prefix outcomes, with hidden targets reserved
for final evaluation. The equations and split-specific rules below specify how
the reported slices and routed policies are evaluated.

## 1. Scope

The protocol applies to the MovieLens 1M and MovieLens 10M user-slate experiments.

Each user is treated as a group. For a sampled view, `k` rated movies are revealed as prefix/context items, and `n` held-out rated movies form the hidden slate. Hidden movie metadata is visible; hidden ratings are used only to compute the evaluation label after the slate has already been selected.

The tail event is

```text
1[ number of hidden movies with rating >= 4 stars >= ceil(c n) ].
```

## 2. Fixed Configurations

Main MovieLens 1M decision-critical configurations:

- `crit40`: `(k, n, c) = (2, 40, 0.7)`.
- `crit80`: `(k, n, c) = (2, 80, 0.7)`.
- `crit80c75`: `(k, n, c) = (2, 80, 0.75)`, used as a harder boundary stress test.

MovieLens 10M scale-up configuration:

- `ml10m_120`: `(k, n, c) = (2, 120, 0.6)` with `max_users = 5000` eligible users.

Random-slate configurations are auxiliary boundary checks, not the headline decision-critical protocol.

## 3. User-Level Splits

Users are split by user identity into train, validation, and test partitions. A test user's ratings never appear in train or validation examples. Model selection and confidence-threshold tuning use validation users. Descriptive slices use unlabeled test-score quantiles; the original switch applies numeric validation cutoffs to test users. The later exact-budget comparison ranks each split separately. Hidden test outcomes never determine selection.

## 4. Allowed Information for Slate Selection

The decision-critical slate selection rule may use only:

- Movie identifiers to index training-set popularity. Genres are prediction-head features, not selector inputs.
- Training-set movie popularity logits or popularity-derived prior success estimates.
- Revealed prefix ratings for the `k` context items.
- Revealed prefix positive/negative bits derived from those observed ratings.
- The fixed configuration `(k, n, c)` and fixed candidate budget.

The selection rule must not use:

- Hidden ratings.
- Hidden high-rating indicators.
- Hidden tail-event labels.
- Future user outcomes.
- Predictions from the posterior head being evaluated.
- Test-label statistics used after slate acceptance.

## 5. Decision-Critical Slate Selection Rule

For each candidate view:

1. Compute the hidden-slate prior success rate `p_hidden_prior` from training-set movie popularity logits indexed by movie identity.
2. Compute the observed prefix success rate `y_obs_bar` using only the `k` revealed prefix ratings.
3. Accept the candidate immediately if:

```text
abs(p_hidden_prior - c) <= 0.08
and the observed prefix contains both a positive and a negative rating outcome.
```

4. If no candidate is accepted immediately, sample up to the fixed candidate budget of `80` and keep the candidate minimizing:

```text
abs(p_hidden_prior - c) + 0.20 * abs(y_obs_bar - 0.5).
```

The constants `0.08`, `0.20`, and `80` are fixed protocol choices. They are not tuned per model.

## 6. TaxScore and Flagged Slices

After selecting a candidate, `grid_predictions` constructs a scalar user-shift posterior on 16 equally spaced points in [-3,3]. Prior weights are proportional to `exp(-theta^2/(2*1.6^2))`, updated with prefix Bernoulli likelihoods using `sigmoid(popularity_logit + theta)`. Hidden-slate component means `mu_g` give `mu_bar = sum w_g mu_g` and `v_hat = sum w_g (mu_g-mu_bar)^2`. The score is:

    S = n * v_hat / (abs(mu_bar - ceil(c*n)/n) + 0.03)^2

This is not the negative candidate boundary score or the exact theoretical two-point TaxScore. The same grid surrogate serves all neural heads and uses no compared-head predictions or hidden targets. The audit reports:

- Full test distribution.
- High-TaxScore bin.
- Top-20% TaxScore-flagged slates.
- Top-30% TaxScore-flagged slates.
- Top-10% TaxScore slice for stress testing.

Descriptive bins use test-score terciles, correlation panels use quartiles, and top-p slices use test-score ranks. Only the original deployment switch transfers numeric validation cutoffs unchanged to test users. The added matched-budget study selects exactly 330/1652 or 496/1652 test slates. Fractions and rules are fixed before evaluation; hidden outcomes do not choose any slice.

## 7. Models Compared

Direct or independently completed predictions:

- Independent marginal completion.
- Direct Transformer count head (`SeqTXCount`).
- Parameter-matched direct count heads.
- Two-tower ensemble mean if ensemble members are averaged at the item-marginal level before tail computation.

Shared-latent or mixture predictions:

- Global grid/mixture heads.
- Transformer or set backbones with explicit shared global-latent heads.
- Attentive neural process latent head.
- Conditional flow latent head.
- Ensemble-mixture slate tails where each ensemble member is propagated across the whole slate before tail probabilities are averaged.

## 8. Metrics

The primary metrics are decision-native:

- Tail-event NLL.
- Tail-event ECE.
- Brier score.
- Accuracy.
- Stop/reveal or triage utility.
- Regret/utility gap.
- Defer or reveal rate where applicable.

Audit metrics:

- TaxScore bin counts.
- Paired seed intervals.
- Signed covariance audit.
- Reliability curves and Brier decomposition.
- Capacity and compute audit.

## 9. Interpretation Rules

The protocol supports the following claims:

- Enrichment of gains on high-score slates is empirical evidence for this routing score, not a universal ranking guarantee or safety certificate.
- If full-distribution utility gains are modest but flagged-slate gains are larger, this supports a budgeted deployment audit rather than a claim that every slate needs a richer posterior head.
- If two-tower or direct count heads have competitive utility but worse NLL/ECE, the conclusion is about tail calibration and posterior interface, not generic ranking accuracy.
- If low-TaxScore or mean-driven tasks show small gains, that is a predicted boundary case, not a contradiction.

The protocol does not support the claim that all sequence models fail. It tests shared dependence for specified tail-sensitive actions. Candidate-robustness sweeps do not perturb every grid-score ingredient; head-only timing excludes scoring and serving overhead.
