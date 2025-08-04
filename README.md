# DSP Multi-Task Prediction Model

This repository provides an end-to-end pipeline for training a multi-task model that simultaneously estimates CTR, CVR, and IVR across various advertising scenarios (geolocation, platform, supply vs. demand).

Key features:

1. **Multi-Task Architecture** – Mixture-of-Experts (MMoE) shared bottom plus task-specific towers.
2. **Scenario Awareness** – Scenario features (fields prefixed with `supply_` and `ip`) are embedded and fed into the gating mechanism so each scenario gets specialized expert weights.
3. **Class-Imbalance Handling** – Per-task positive-class weighting and optional focal loss.
4. **Dynamic Loss Balancing** – Uncertainty-based weighting automatically adjusts the contribution of each task.
5. **Metric Calibration** – Post-training Platt scaling to align predicted probabilities with true outcome rates.
6. **Production Ready** – Early stopping, model checkpointing, configurable hyper-parameters, and inference script.

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Format
Input data should be a CSV/Parquet with the following columns:

* Sparse categorical: `hour, weekday, adv_id, affiliate_id, campaign_id, ad_group_id, ad_id, creative_id, feature_1, pos, instl, response_type, ad_format, os, device_make, bundle_id, country, package, category, connection_type, device_model, lang, publisher_id, first_ssp, last_ssp, video_placement, is_rewarded, offer_id, supply_developer_id, supply_genreId, supply_version, supply_minimum_os_version, supply_industry_id, is_oem, tag_id, osv, ua, demand_developer_id, demand_genreId, demand_version, demand_minimum_os_version, demand_industry_id, ip, device_id, ad_width, ad_height`
* Label columns: `ctr_label, cvr_label, ivr_label` (1 for positive, 0 for negative)

## Quick Start
```bash
python train.py --data_path /path/to/data.parquet --gpus 1 --max_epochs 10
```

Detailed configuration parameters can be found inside `train.py`.
