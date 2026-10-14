import argparse
import json
import os
from os.path import join

import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from config import DATA_DIR, REPO_DIR
from encoding_utils import apply_zscore_and_hrf, get_response, get_story_split
from whisper_feature_spaces import get_whisper_decoder_feature_space

def to_numpy(array):
    """Move backend results to host memory before NumPy operations or saving."""
    if hasattr(array, "get"):
        return array.get()
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "numpy"):
        return array.numpy(force=True)
    return np.asarray(array)


def configure_himalaya_backend(backend="torch_cuda", cpu_threads=1):
    if cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1.")

    set_backend(backend)
    threadpool_limits(limits=cpu_threads)

    if backend == "torch_cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The torch_cuda backend was requested, but CUDA is unavailable. "
                "Use --backend numpy for CPU execution."
            )
        torch.set_num_threads(cpu_threads)

    print(f"Himalaya backend: {backend}")
    print(f"CPU threads: {cpu_threads}")


class LiveLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def __call__(self, msg=""):
        print(msg)
        with open(self.filepath, "a") as f:
            f.write(str(msg) + "\n")
            f.flush()

def select_optimal_decoder_layer(
    subjects,
    train_stories,
    modelname,
    candidate_layers,
    feature_dir=None,
    trim=5,
    ndelays=4,
    use_corr=True,
    backend="torch_cuda",
    cpu_threads=1,
):
    configure_himalaya_backend(backend=backend, cpu_threads=cpu_threads)

    evals_dir = join(DATA_DIR, "derivatives", "results", {modelname}, "layer_evals")
    os.makedirs(evals_dir, exist_ok=True)
    
    if len(candidate_layers) == 1:
        single_layer = candidate_layers[0]
        log_file_path = join(evals_dir, f"layer_{single_layer}_output.txt")
    else:
        log_file_path = join(evals_dir, "layer_selection_summary.txt")

    log = LiveLogger(log_file_path)

    log("=" * 80)
    log(f"DECISION LOG: LAYER SELECTION RUN ({modelname})")
    log("=" * 80)
    log(f"Subjects Used:     {subjects}")
    log(f"Training Stories: {train_stories}")
    log(f"Candidate Layers: {candidate_layers}")
    log(f"Parameters:       trim={trim}, ndelays={ndelays}, use_corr={use_corr}")
    log(f"Output File:      {log_file_path}")
    log("-" * 80)

    if len(train_stories) < 2:
        raise ValueError("At least two training stories required.")
    if not subjects:
        raise ValueError("At least one subject required.")

    alphas = np.logspace(0, 3, 10)
    layer_scores_by_subject = {layer: [] for layer in candidate_layers}
    cv_splitter = KFold(n_splits=5, shuffle=False)

    for subject in subjects:
        log(f"\nProcessing Subject: {subject}")
        log("-" * 40)

        _, good_stories = get_response(train_stories, subject)

        if len(good_stories) < 2:
            log(f"  WARNING: Only {len(good_stories)} valid stories for {subject}, skipping")
            continue

        log(f"  Using {len(good_stories)} valid training stories")

        for layer in candidate_layers:
            
            feature_space = get_whisper_decoder_feature_space(
                good_stories,
                modelname=modelname,
                layer=layer,
                feature_dir=feature_dir,
            )

            full_stim_raw = apply_zscore_and_hrf(
                good_stories, feature_space, trim, ndelays
            ).astype(np.float32)
            full_stim_raw = np.nan_to_num(full_stim_raw, nan=0.0, posinf=0.0, neginf=0.0)

            full_resp, _ = get_response(good_stories, subject)
            valid_voxel_mask = np.isfinite(full_resp).all(axis=0)
            full_resp_masked = full_resp[:, valid_voxel_mask].astype(np.float32)

            story_row_bounds = {}
            current_row = 0
            for story in good_stories:
                story_feat = feature_space[story]
                n_trs = story_feat.shape[0] - (2 * trim)
                story_row_bounds[story] = (current_row, current_row + n_trs)
                current_row += n_trs

            layer_validation_scores = []

            for validation_story in good_stories:
                log(f"     Starting story: {validation_story}")
                val_start, val_end = story_row_bounds[validation_story]

                val_row_mask = np.zeros(full_stim_raw.shape[0], dtype=bool)
                val_row_mask[val_start:val_end] = True
                train_row_mask = ~val_row_mask

                inner_train_stim_raw = full_stim_raw[train_row_mask]
                validation_stim_raw = full_stim_raw[val_row_mask]

                inner_train_resp_masked = full_resp_masked[train_row_mask]
                validation_resp_masked = full_resp_masked[val_row_mask]

                scaler = StandardScaler(with_mean=True, with_std=False)
                inner_train_stim_scaled = scaler.fit_transform(inner_train_stim_raw)
                validation_stim_scaled = scaler.transform(validation_stim_raw)

                log(f"     Fitting model with KernelRidgeCV")
                model = KernelRidgeCV(
                    alphas=alphas,
                    cv=cv_splitter,
                    solver="eigenvalues",
                )
                model.fit(inner_train_stim_scaled, inner_train_resp_masked)

                validation_pred_backend = model.predict(validation_stim_scaled)

                if use_corr:
                    validation_corr = correlation_score_split(
                        validation_resp_masked, validation_pred_backend
                    )
                    validation_corr = to_numpy(validation_corr)
                else:
                    validation_pred_masked = to_numpy(validation_pred_backend)
                    residual_var = np.var(validation_resp_masked - validation_pred_masked, axis=0)
                    resp_var = np.var(validation_resp_masked, axis=0)
                    validation_corr = np.sqrt(
                        np.abs(1.0 - residual_var / resp_var)
                    ) * np.sign(1.0 - residual_var / resp_var)

                validation_score = float(np.nanmean(validation_corr))
                layer_validation_scores.append(validation_score)

            mean_layer_score = float(np.nanmean(layer_validation_scores))
            layer_scores_by_subject[layer].append(mean_layer_score)
            log(f"  Layer {layer}: {mean_layer_score:.6f}")
    
    # Write output to JSONs
    # Save single layer output to JSON
    if len(candidate_layers) == 1:
        single_layer = candidate_layers[0]
        layer_eval_data = {
            "layer": int(single_layer),
            "subjects": subjects,
            "scores_by_subject": layer_scores_by_subject[single_layer],
            "mean_score": float(np.nanmean(layer_scores_by_subject[single_layer])),
        }
        eval_path = join(evals_dir, f"layer_{single_layer}_results.json")
        with open(eval_path, "w") as f:
            json.dump(layer_eval_data, f, indent=2)
        log(f"Saved layer {single_layer} evaluation to: {eval_path}")
        return single_layer, layer_eval_data

    # Aggregator Mode: Collect all per-layer evaluation JSONs
    log("\n" + "=" * 80)
    log("POOLED LAYER SELECTION RESULTS")
    log("=" * 80)

    means = {}
    standard_errors = {}

    for layer in candidate_layers:
        eval_path = join(evals_dir, f"layer_{layer}_results.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            layer_scores_by_subject[layer] = eval_data["scores_by_subject"]

        if layer_scores_by_subject[layer]:
            scores = np.array(layer_scores_by_subject[layer])
            mean_score = float(np.nanmean(scores))
            se = float(
                np.nanstd(scores, ddof=1) / np.sqrt(len(scores))
                if len(scores) > 1
                else 0.0
            )
            means[layer] = mean_score
            standard_errors[layer] = se
            log(f"Layer {layer:2d}: Pooled Mean = {mean_score:.6f} +/- {se:.6f}")
        else:
            log(f"Layer {layer:2d}: SKIPPED (no valid scores found)")

    if not means:
        raise ValueError("No valid layer scores obtained from layer_evals directory.")

    best_layer = max(means, key=means.get)
    best_mean = means[best_layer]
    best_se = standard_errors[best_layer]

    threshold = best_mean - best_se
    acceptable_layers = [
        layer for layer, score in means.items() if score >= threshold
    ]
    selected_layer = max(acceptable_layers)

    log("-" * 80)
    log(f"Best validation layer:          {best_layer} (Score: {best_mean:.6f})")
    log(f"Selection 1-SE threshold:       {threshold:.6f}")
    log(f"Acceptable layers within 1-SE: {acceptable_layers}")
    log(f"SELECTED OPTIMAL LAYER:         {selected_layer}")
    log("=" * 80 + "\n")

    layer_selection = {
        "candidate_layers": [int(layer) for layer in candidate_layers],
        "subjects_used": subjects,
        "means": {int(k): v for k, v in means.items()},
        "standard_errors": {int(k): v for k, v in standard_errors.items()},
        "scores": {
            int(layer): scores
            for layer, scores in layer_scores_by_subject.items()
            if scores
        },
        "best_layer": int(best_layer),
        "selected_layer": int(selected_layer),
        "selection_threshold": float(threshold),
        "selection_method": "leave-one-story-out with Kernel RidgeCV",
        "use_corr": bool(use_corr),
    }

    save_location = join(REPO_DIR, "results", "whisper")
    os.makedirs(save_location, exist_ok=True)
    layer_selection_path = join(save_location, "layer_selection.json")

    with open(layer_selection_path, "w") as f:
        json.dump(layer_selection, f, indent=2)
    log(f"Saved layer selection metadata to: {layer_selection_path}")

    return selected_layer, layer_selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["UTS01", "UTS02", "UTS03"])
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--modelname", type=str, default="whisper")
    parser.add_argument("--layer", type=int, default=None, help="Single layer for SLURM array execution")
    parser.add_argument("--optimal-start", type=int, default=22)
    parser.add_argument("--optimal-stop", type=int, default=32)
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--use-corr", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["torch_cuda", "cupy", "numpy"],
        default="torch_cuda",
        help="Himalaya backend. Defaults to PyTorch CUDA for GPU runtimes.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="CPU thread limit for preprocessing and numerical libraries.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Cap maximum training stories used during layer selection for speed",
    )
    args = parser.parse_args()
    args = parser.parse_args()

    train_stories, _ = get_story_split(args.sessions)
    
    if args.max_stories is not None and args.max_stories > 0:
        train_stories = train_stories[: args.max_stories]

    if args.layer is not None:
        candidate_layers = [args.layer]
    else:
        candidate_layers = list(range(args.optimal_start, args.optimal_stop + 1))

    select_optimal_decoder_layer(
        subjects=args.subjects,
        train_stories=train_stories,
        modelname=args.modelname,
        candidate_layers=candidate_layers,
        feature_dir=args.feature_dir,
        trim=args.trim,
        ndelays=args.ndelays,
        use_corr=args.use_corr,
        backend=args.backend,
        cpu_threads=args.cpu_threads,
    )