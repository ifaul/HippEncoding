import os
import argparse
import hashlib
import json
import logging
import multiprocessing
from os.path import join
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV, KernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from config import DATA_DIR
from encoding_utils import apply_zscore_and_hrf, get_response, get_story_split
from feature_spaces import FEATURE_DIR, get_feature_space as get_model_feature_space


def to_numpy(array):
    """Move backend results to host memory before NumPy operations or saving."""
    if hasattr(array, "get"):
        return array.get()
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "numpy"):
        return array.numpy(force=True)
    return np.asarray(array)


def configure_himalaya_backend(backend="torch_mps", n_jobs=1, omp_threads=1):
    set_backend(backend)
    if omp_threads < 1:
        raise ValueError("--omp-threads must be at least 1.")

    threadpool_limits(limits=omp_threads)
    if backend == "torch_mps":
        import torch

        torch.set_num_threads(omp_threads)

    effective_n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    return effective_n_jobs


def run_encoding(args):
    modelname = args.modelname.lower()
    if "whisper" in modelname:
        model_family = "whisper"
    elif modelname == "gpt2-xl":
        model_family = "gpt"
    else:
        raise ValueError(
            f"Unsupported model '{args.modelname}'. "
            "Supported models are Whisper models and gpt2-xl."
        )

    effective_n_jobs = configure_himalaya_backend(
        backend=args.backend,
        n_jobs=args.n_jobs,
        omp_threads=args.omp_threads,
    )
    
    if not args.sessions:
        raise ValueError("At least one session must be supplied.")

    if min(args.sessions) < 1 or max(args.sessions) > 5:
        raise ValueError("Sessions must satisfy 1 <= session <= 5.")

    train_stories, test_stories = get_story_split(args.sessions)
    
    if args.max_stories is not None and args.max_stories > 0:
            train_stories = train_stories[: args.max_stories]

    # Get valid stories from fMRI data
    zRresp, response_train_stories = get_response(train_stories, args.subject)
    print("Loaded fMRI data for training stories")
    zPresp, response_test_stories = get_response(test_stories, args.subject)
    print("Loaded fMRI data for testing stories")
    valid_train_stories = response_train_stories
    valid_test_stories = response_test_stories
    all_stories = list(dict.fromkeys(valid_train_stories + valid_test_stories))

    # Select a decoder layer and load model features.
    selected_layer = args.layer

    if model_family == "whisper":
        layer_selection_path = join(
            DATA_DIR, "derivatives", "results", args.modelname, "layer_selection.json"
        )
        if os.path.exists(layer_selection_path):
            with open(layer_selection_path, "r") as f:
                layer_selection = json.load(f)
            selected_layer = layer_selection["selected_layer"]
            logging.info("Loaded cached optimal layer: %s", selected_layer)
        elif selected_layer is None:
            raise ValueError(
                "Provide --layer or run layer selection before encoding Whisper."
            )
        include_conv = True
        include_enc = True
    else:
        if args.context_len < 1:
            raise ValueError("--context-len must be at least 1 for gpt2-xl.")
        if selected_layer is None:
            raise ValueError("--layer is required when using gpt2-xl.")
        include_conv = False
        include_enc = False

    downsampled_feat, _ = get_model_feature_space(
        all_stories,
        modelname=args.modelname,
        context_len=args.context_len if model_family == "gpt" else None,
        layer=selected_layer,
        include_conv=include_conv,
        include_enc=include_enc,
    )
    print("Loaded feature spaces for all stories")

    feature_stories = set(downsampled_feat)
    valid_train_stories = [
        story for story in valid_train_stories if story in feature_stories
    ]
    valid_test_stories = [
        story for story in valid_test_stories if story in feature_stories
    ]
    if not valid_train_stories or not valid_test_stories:
        raise ValueError(
            f"No stories have both response and {args.modelname} features "
            f"for subject {args.subject}."
        )

    if valid_train_stories != response_train_stories:
        zRresp, _ = get_response(valid_train_stories, args.subject)
    if valid_test_stories != response_test_stories:
        zPresp, _ = get_response(valid_test_stories, args.subject)
    
    save_location = join(DATA_DIR, "derivatives", "results", args.modelname)
    if model_family == "gpt":
        save_location = join(save_location, f"W_{args.context_len}")
    save_location = join(save_location, args.subject)
    
    print("Stimulus and response parameters:")
    print(f"trim-start: {args.trim_start}, trim-end: {args.trim_end}, ndelays: {args.ndelays}")

    cache_parameters = {
        "train_stories": valid_train_stories,
        "test_stories": valid_test_stories,
        "layer": selected_layer,
        "context_len": args.context_len if model_family == "gpt" else None,
        "trim-start": args.trim_start,
        "trim-end": args.trim_end,
        "ndelays": args.ndelays,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_parameters, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    feature_cache_dir = Path(FEATURE_DIR, args.modelname)
    if model_family == "gpt":
        feature_cache_dir /= f"W_{args.context_len}"
    cache_dir = feature_cache_dir / "encoding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    delRstim_path = cache_dir / f"delRstim_{cache_key}.npy"
    delPstim_path = cache_dir / f"delPstim_{cache_key}.npy"

    if delRstim_path.is_file(): 
        print("Loading saved delRstim")
        delRstim = np.load(delRstim_path)
    else:
        print("Applying z-scoring and hrf to training stimulus")
        delRstim = apply_zscore_and_hrf(valid_train_stories, downsampled_feat, args.trim_start, args.trim_end, args.ndelays).astype(np.float32)
        np.save(delRstim_path, delRstim)

    if delPstim_path.is_file(): 
        print("Loading saved delPstim")
        delPstim = np.load(delPstim_path)
    else:
        print("Applying z-scoring and hrf to testing stimulus")
        delPstim = apply_zscore_and_hrf(valid_test_stories, downsampled_feat, args.trim_start, args.trim_end, args.ndelays).astype(np.float32)
        np.save(delPstim_path, delPstim)
    
    print("delRstim shape:", delRstim.shape)
    print("delPstim shape:", delPstim.shape)

    # Filter by non-nan rows
    train_valid = np.isfinite(zRresp).all(axis=0)
    test_valid = np.isfinite(zPresp).all(axis=0)
    valid_voxel_mask = train_valid & test_valid

    # Mask response matrices
    zRresp_masked = zRresp[:, valid_voxel_mask].astype(np.float32)
    zPresp_masked = zPresp[:, valid_voxel_mask].astype(np.float32)
    print("zRresp_masked shape:", zRresp_masked.shape)
    print("zPresp_masked shape:", zPresp_masked.shape)

    if delRstim.shape[0] != zRresp_masked.shape[0]:
        raise ValueError(f"Training mismatch: {delRstim.shape[0]} vs {zRresp.shape[0]}")

    if delPstim.shape[0] != zPresp_masked.shape[0]:
        raise ValueError(f"Test mismatch: {delPstim.shape[0]} vs {zPresp.shape[0]}")

    # =========================================================================
    # Construct Explicit Feature Indices Across HRF Delays
    # =========================================================================
    total_undelayed_features = downsampled_feat[valid_train_stories[0]].shape[1]
    n_features_total = delRstim.shape[1]
    if total_undelayed_features == 0 or n_features_total % total_undelayed_features:
        raise ValueError(
            "Delayed stimulus feature count is not a multiple of the "
            f"undelayed feature count ({n_features_total} vs "
            f"{total_undelayed_features})."
        )
    n_delays_count = n_features_total // total_undelayed_features

    if model_family == "whisper":
        if args.modelname == "whisper-large":
            n_acoustic = n_speech = n_language = 1280
        elif args.modelname == "whisper-medium":
            n_acoustic = n_speech = n_language = 1024
        else:
            n_acoustic = n_speech = n_language = total_undelayed_features // 3

        if n_acoustic + n_speech + n_language != total_undelayed_features:
            raise ValueError(
                f"Unexpected Whisper feature width {total_undelayed_features} "
                f"for model {args.modelname}."
            )
        feature_slices = {
            "acoustic": np.concatenate([
                np.arange(d * total_undelayed_features,
                          d * total_undelayed_features + n_acoustic)
                for d in range(n_delays_count)
            ]),
            "speech": np.concatenate([
                np.arange(d * total_undelayed_features + n_acoustic,
                          d * total_undelayed_features + n_acoustic + n_speech)
                for d in range(n_delays_count)
            ]),
            "language": np.concatenate([
                np.arange(d * total_undelayed_features + n_acoustic + n_speech,
                          (d + 1) * total_undelayed_features)
                for d in range(n_delays_count)
            ]),
        }
    else:
        feature_slices = {
            "language": np.concatenate([
                np.arange(
                    d * total_undelayed_features,
                    (d + 1) * total_undelayed_features,
                )
                for d in range(n_delays_count)
            ])
        }

    print("\nFeature spaces setup (across HRF delays):")
    for name, idxs in feature_slices.items():
        print(f"  {name}: {len(idxs)} features")

    alphas = np.logspace(1, 30, 30, dtype=np.float32)
    diagonalize_method = (args.diagonalize_method)
    n_alphas_batch = getattr(args, "n_alphas_batch", 1)
    n_iter = getattr(args, "n_iter", 30)
    print(f"Kernel diagonalization method: {diagonalize_method}")

    # =========================================================================
    # 1. Joint Banded Ridge Model (acoustic + speech + language)
    # =========================================================================
    run_joint_model = (
        model_family == "whisper" and args.featurespace != "individual"
    )
    run_individual_models = (
        args.featurespace != "joint" or model_family == "gpt"
    )

    if run_joint_model:
        acoustic_indices = feature_slices["acoustic"]
        speech_indices = feature_slices["speech"]
        language_indices = feature_slices["language"]

        print("\nFitting joint Banded Ridge model (acoustic + speech + language)...")
        
        joint_save_location = os.path.join(save_location, "..", "joint", args.subject)
        os.makedirs(joint_save_location, exist_ok=True)
        
        # 3 separate kernels allow MultipleKernelRidgeCV to optimize alphas per feature space
        joint_kernelizer = ColumnKernelizer(
            [
                ("acoustic", Kernelizer(kernel="linear"), acoustic_indices),
                ("speech", Kernelizer(kernel="linear"), speech_indices),
                ("language", Kernelizer(kernel="linear"), language_indices),
            ],
            n_jobs=effective_n_jobs,
        )
        
        joint_params = dict(
            alphas=alphas,
            progress_bar=True,
            diagonalize_method=args.diagonalize_method,
            n_iter=n_iter,
            n_alphas_batch=args.n_alphas_batch,
            n_targets_batch=args.n_targets_batch,
        )
        
        joint_model = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            joint_kernelizer,
            MultipleKernelRidgeCV(kernels="precomputed", solver_params=joint_params),
        )
        
        joint_model.fit(delRstim, zRresp_masked)
        joint_pred_backend = joint_model.predict(delPstim)
        joint_corrs = to_numpy(correlation_score_split(zPresp_masked, joint_pred_backend))
        joint_pred = to_numpy(joint_pred_backend)
        
        print(f"Joint model mean correlation: {np.nanmean(joint_corrs):.6f}")
        
        joint_ridge = joint_model[-1]

        # Extract primal weights using StandardScaler-transformed features
        delRstim_scaled = to_numpy(joint_model[0].transform(delRstim))

        try:
            X_joint_list = [
                delRstim_scaled[:, acoustic_indices],
                delRstim_scaled[:, speech_indices],
                delRstim_scaled[:, language_indices],
            ]
            joint_weights = to_numpy(joint_ridge.get_primal_coef(X_joint_list))
        except Exception as e:
            print(f"Note: Storing dual coefficients for joint model ({e}).")
            dual_coef = getattr(joint_ridge, "dual_coef_", None)
            joint_weights = to_numpy(dual_coef) if dual_coef is not None else None

        # Reconstruct full-length arrays matching original voxel dimensions
        total_voxels = zRresp.shape[1]

        # Full Correlation Vector (Total_Voxels,)
        full_corrs = np.full(total_voxels, np.nan, dtype=np.float32)
        full_corrs[valid_voxel_mask] = joint_corrs

        # Full Predictions Matrix (Test_TRs, Total_Voxels)
        full_pred = np.full_like(zPresp, np.nan, dtype=np.float32)
        full_pred[:, valid_voxel_mask] = joint_pred

        # Full Weights Matrix (Features, Total_Voxels)
        if joint_weights is not None:
            n_features = joint_weights.shape[0]
            full_weights = np.full((n_features, total_voxels), np.nan, dtype=np.float32)
            full_weights[:, valid_voxel_mask] = joint_weights
        else:
            full_weights = None

        # Save full-dimensional arrays to disk
        np.savez(os.path.join(joint_save_location, "corrs.npz"), corrs=full_corrs)
        np.savez(os.path.join(joint_save_location, "predictions.npz"), predictions=full_pred)

        if full_weights is not None:
            np.savez(os.path.join(joint_save_location, "weights.npz"), weights=full_weights)

        if hasattr(joint_ridge, "best_alphas_"):
            np.savez(
                os.path.join(joint_save_location, "alphas.npz"),
                alphas=to_numpy(joint_ridge.best_alphas_),
            )
        
        metadata_joint = {
            "subject": args.subject,
            "feature": model_family,
            "modelname": args.modelname,
            "layer": selected_layer,
            "sessions": args.sessions,
            "train_stories": valid_train_stories,
            "test_stories": valid_test_stories,
            "model_type": "joint",
            "feature_spaces": ["acoustic", "speech", "language"],
            "mean_correlation": float(np.nanmean(joint_corrs)),
        }
        
        with open(os.path.join(joint_save_location, "metadata.json"), "w") as f:
            json.dump(metadata_joint, f, indent=2)

    # =========================================================================
    # 2. Individual Feature Space Models
    # =========================================================================
    if run_individual_models:
        individual_results = {}
        
        for space_name, space_indices in tqdm(
            feature_slices.items(),
            desc="Fitting individual feature space models",
            unit="space",
        ):
            print(f"\nFitting {space_name} individual model...")
            
            individual_save_location = os.path.join(save_location, "..", space_name, args.subject)
            os.makedirs(individual_save_location, exist_ok=True)
            
            space_params = dict(
                diagonalize_method=diagonalize_method,
                n_targets_batch=args.n_targets_batch,
                n_alphas_batch=n_alphas_batch,
            )

            train_features = delRstim[:, space_indices]
            test_features = delPstim[:, space_indices]

            space_model = make_pipeline(
                StandardScaler(with_mean=True, with_std=False),
                Kernelizer(kernel="linear"),
                KernelRidgeCV(
                    alphas=alphas,
                    kernel="precomputed",
                    solver_params=space_params,
                ),
            )
            
            # 1. Fit & Predict using MASKED responses
            space_model.fit(train_features, zRresp_masked)
            space_pred_backend = space_model.predict(test_features)
            space_corrs_masked = to_numpy(
                correlation_score_split(zPresp_masked, space_pred_backend)
            )
            space_pred_masked = to_numpy(space_pred_backend)
            
            print(f"{space_name} model mean correlation: {np.nanmean(space_corrs_masked):.6f}")
            
            space_ridge = space_model[-1]
            
            # 2. Extract primal weights using StandardScaler-transformed features
            delRstim_scaled = to_numpy(space_model[0].transform(train_features))

            try:
                # Pass 2D feature matrix directly (n_samples, n_features)
                space_weights = to_numpy(
                    space_ridge.get_primal_coef(delRstim_scaled)
                )
            except Exception as e:
                print(f"Note: Storing dual coefficients for {space_name} ({e}).")
                dual_coef = getattr(space_ridge, "dual_coef_", None)
                space_weights = to_numpy(dual_coef) if dual_coef is not None else None

            # 3. Reconstruct full-length arrays matching original voxel dimensions
            total_voxels = zRresp.shape[1]

            # Full Correlation Vector (Total_Voxels,)
            full_corrs = np.full(total_voxels, np.nan, dtype=np.float32)
            full_corrs[valid_voxel_mask] = space_corrs_masked

            # Full Predictions Matrix (Test_TRs, Total_Voxels)
            full_pred = np.full_like(zPresp, np.nan, dtype=np.float32)
            full_pred[:, valid_voxel_mask] = space_pred_masked

            # Full Weights Matrix (Space_Features, Total_Voxels)
            if space_weights is not None:
                n_features_space = space_weights.shape[0]
                full_weights = np.full((n_features_space, total_voxels), np.nan, dtype=np.float32)
                full_weights[:, valid_voxel_mask] = space_weights
            else:
                full_weights = None

            # Save full-dimensional arrays to disk
            np.savez(os.path.join(individual_save_location, "corrs.npz"), corrs=full_corrs)
            np.savez(os.path.join(individual_save_location, "predictions.npz"), predictions=full_pred)

            if full_weights is not None:
                np.savez(os.path.join(individual_save_location, "weights.npz"), weights=full_weights)

            if hasattr(space_ridge, "best_alphas_"):
                np.savez(
                    os.path.join(individual_save_location, "alphas.npz"),
                    alphas=to_numpy(space_ridge.best_alphas_),
                )

            metadata_individual = {
                "subject": args.subject,
                "feature": model_family,
                "modelname": args.modelname,
                "layer": int(selected_layer),
                "sessions": args.sessions,
                "train_stories": valid_train_stories,
                "test_stories": valid_test_stories,
                "model_type": space_name,
                "mean_correlation": float(np.nanmean(space_corrs_masked)),
                "weights_saved": full_weights is not None,
            }

            with open(os.path.join(individual_save_location, "metadata.json"), "w") as f:
                json.dump(metadata_individual, f, indent=2)

            individual_results[space_name] = {
                "corrs": full_corrs,
                "predictions": full_pred,
                "weights": full_weights,
            }

    if model_family == "whisper" and args.featurespace == "both":
        # Summary Output
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Joint model (all features): {np.nanmean(joint_corrs):.6f}")
        for space_name, results in individual_results.items():
            print(f"{space_name.capitalize():12s} model:        {np.nanmean(results['corrs']):.6f}")
        print("=" * 80)

def _run_subject(args, subject):
    subject_args = argparse.Namespace(**vars(args))
    subject_args.subject = subject
    run_encoding(subject_args)


def run_subjects(args):
    subjects = args.subjects if args.subjects else [args.subject]
    if args.subject_jobs < 1:
        raise ValueError("--subject-jobs must be at least 1.")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subjects must be unique to avoid concurrent output collisions.")

    worker_count = min(args.subject_jobs, len(subjects))
    if worker_count == 1:
        for subject in subjects:
            _run_subject(args, subject)
        return

    print(
        f"Running {len(subjects)} subjects across {worker_count} processes "
        f"with backend {args.backend}. Each process uses the configured GPU."
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(_run_subject, args, subject)
            for subject in subjects
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument("--subject", type=str)
    subject_group.add_argument(
        "--subjects",
        nargs="+",
        help="Subjects to encode. Use --subject-jobs to run them in parallel.",
    )
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2, 3, 4, 5])

    parser.add_argument(
        "--modelname",
        type=str,
        default="whisper-large",
        help="Whisper model name or gpt2-xl.",
    )
    parser.add_argument("--featurespace", type=str, default="both", choices=["joint","individual","both"])
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Decoder layer to encode; required for GPT-2 XL.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
        help="GPT-2 XL context horizon used to locate saved features.",
    )
    parser.add_argument(
                "--max-stories",
                type=int,
                default=None,
                help="Cap maximum training stories used during layer selection for speed",
            )

    # Feature preprocessing
    parser.add_argument("--trim_start", type=int, default=50)
    parser.add_argument("--trim_end", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    
    # Himalaya backend parameters
    parser.add_argument(
        "--backend",
        type=str,
        default="torch_mps",
        choices=["numpy", "cupy", "torch_cuda", "torch_mps"],
        help="Himalaya backend. Defaults to PyTorch CUDA for GPU runtimes.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel kernelizer jobs (-1 uses all detected CPUs).",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=1,
        help="CPU thread limit per process (also applies to PyTorch intra-op threads).",
    )
    parser.add_argument(
        "--n-targets-batch",
        type=int,
        default=5000,
        help=(
            "Voxel/target batch size for Himalaya CV solvers. Smaller values "
            "reduce accelerator memory use"
        ),
    )
    parser.add_argument(
        "--n-alphas-batch",
        type=int,
        default=1,
        help=(
            "Number of regularization values to process at once. Smaller values "
            "reduce temporary memory use."
        ),
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of random kernel-weight candidates in joint-model CV.",
    )
    parser.add_argument(
        "--diagonalize-method",
        choices=["eigh", "svd"],
        default="svd",
        help=(
            "Kernel decomposition for symmetric Gram matrices; eigh is preferred."
        ),
    )
    parser.add_argument(
        "--subject-jobs",
        type=int,
        default=1,
        help=(
            "Number of subject processes. Each process initializes its own GPU "
            "context and model, so increase only when GPU memory permits."
        ),
    )

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    if args.n_targets_batch < 1:
        parser.error("--n-targets-batch must be at least 1.")
    if args.n_alphas_batch < 1:
        parser.error("--n-alphas-batch must be at least 1.")
    if args.n_iter < 1:
        parser.error("--n-iter must be at least 1.")
    run_subjects(args)