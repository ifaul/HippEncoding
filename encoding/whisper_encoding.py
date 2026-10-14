import os
import argparse
import json
import logging
import multiprocessing
from os.path import join
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV, KernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from config import EM_DATA_DIR, REPO_DIR
from encoding_utils import apply_zscore_and_hrf, get_response, get_story_split
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from whisper_feature_spaces import get_whisper_feature_space, get_whisper_decoder_feature_space, get_decoder_layer_count


def to_numpy(array):
    """Move backend results to host memory before NumPy operations or saving."""
    if hasattr(array, "get"):
        return array.get()
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "numpy"):
        return array.numpy(force=True)
    return np.asarray(array)


def configure_himalaya_backend(backend="torch_cuda", n_jobs=1, omp_threads=1):
    set_backend(backend)
    if omp_threads < 1:
        raise ValueError("--omp-threads must be at least 1.")

    threadpool_limits(limits=omp_threads)
    if backend == "torch_cuda":
        import torch

        torch.set_num_threads(omp_threads)

    effective_n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    print(f"Himalaya backend: {backend}")
    print(f"Kernelizer jobs: {effective_n_jobs}")
    print(f"CPU threads per process: {omp_threads}")
    return effective_n_jobs


def run_encoding(args):
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
    all_stories = list(dict.fromkeys(train_stories + test_stories))

    selected_layer = args.layer

    if args.feature == "whisper":
        
        save_location = join(REPO_DIR, "results", args.feature)
        layer_selection_path = join(save_location, "layer_selection.json")
        
        if os.path.exists(layer_selection_path):
            with open(layer_selection_path, "r") as f:
                layer_selection = json.load(f)
                
            selected_layer = layer_selection["selected_layer"]
            logging.info("Loaded cached optimal layer: %s", selected_layer)
        elif selected_layer in range(1, 33):
            selected_layer = selected_layer
        else:
            raise Exception("Need to add layer arg or run layer_selection script before running encoding")
        
        downsampled_feat, _ = get_whisper_feature_space(
            all_stories,
            modelname=args.modelname,
            layer=selected_layer,
            feature_dir=args.feature_dir,
        )

    else:
        if args.feature not in _FEATURE_CONFIG:
            available = " ".join(_FEATURE_CONFIG.keys())
            raise ValueError(f"Unknown feature '{args.feature}'. Available: {available}")

        downsampled_feat = get_feature_space(args.feature, all_stories)
    
    save_location = join(REPO_DIR, "results", args.feature, args.subject)
    os.makedirs(save_location, exist_ok=True)
    
    print("Stimulus and response parameters:")
    print(f"trim: {args.trim}, ndelays: {args.ndelays}")

    delRstim = apply_zscore_and_hrf(train_stories, downsampled_feat, args.trim, args.ndelays).astype(np.float32)
    delPstim = apply_zscore_and_hrf(test_stories, downsampled_feat, args.trim, args.ndelays).astype(np.float32)
    print("delRstim shape:", delRstim.shape)
    print("delPstim shape:", delPstim.shape)

    zRresp, valid_train_stories = get_response(train_stories, args.subject)
    zPresp, valid_test_stories = get_response(test_stories, args.subject)

    # Filter by non-nan rows
    train_valid = np.isfinite(zRresp).all(axis=0)
    test_valid = np.isfinite(zPresp).all(axis=0)
    valid_voxel_mask = train_valid & test_valid

    # Mask response matrices
    zRresp_masked = zRresp[:, valid_voxel_mask].astype(np.float32)
    zPresp_masked = zPresp[:, valid_voxel_mask].astype(np.float32)
    print("zRresp_masked shape:", zRresp_masked.shape)
    print("zPresp_masked shape:", zPresp_masked.shape)

    train_stories = valid_train_stories
    test_stories = valid_test_stories

    if delRstim.shape[0] != zRresp.shape[0]:
        raise ValueError(f"Training mismatch: {delRstim.shape[0]} vs {zRresp.shape[0]}")

    if delPstim.shape[0] != zPresp.shape[0]:
        raise ValueError(f"Test mismatch: {delPstim.shape[0]} vs {zPresp.shape[0]}")

    # =========================================================================
    # Construct Explicit Feature Indices Across HRF Delays
    # =========================================================================
    total_undelayed_features = downsampled_feat[train_stories[0]].shape[1]
    
    if args.feature == "whisper":
        n_acoustic = n_speech = n_language = 1280
    else:
        n_acoustic = n_speech = n_language = total_undelayed_features // 3

    n_features_total = delRstim.shape[1]
    n_delays_count = n_features_total // total_undelayed_features

    # Map indices cleanly across delay blocks [D0, D1, D2, D3]
    acoustic_indices = np.concatenate([
        np.arange(d * total_undelayed_features, d * total_undelayed_features + n_acoustic)
        for d in range(n_delays_count)
    ])
    speech_indices = np.concatenate([
        np.arange(d * total_undelayed_features + n_acoustic, d * total_undelayed_features + n_acoustic + n_speech)
        for d in range(n_delays_count)
    ])
    language_indices = np.concatenate([
        np.arange(d * total_undelayed_features + n_acoustic + n_speech, (d + 1) * total_undelayed_features)
        for d in range(n_delays_count)
    ])

    feature_slices = {
        "acoustic": acoustic_indices,
        "speech": speech_indices,
        "language": language_indices,
    }

    print("\nFeature spaces setup (across HRF delays):")
    for name, idxs in feature_slices.items():
        print(f"  {name}: {len(idxs)} features")

    alphas = np.logspace(0, 3, 10)

    # =========================================================================
    # 1. Joint Banded Ridge Model (acoustic + speech + language)
    # =========================================================================
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
        n_iter=100,
        diagonalize_method="eigh",
        return_alphas=True,
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
        np.savez(os.path.join(joint_save_location, "alphas.npz"), alphas=joint_ridge.best_alphas_)
    
    metadata_joint = {
        "subject": args.subject,
        "feature": args.feature,
        "modelname": args.modelname,
        "layer": selected_layer if args.feature == "whisper" else None,
        "sessions": args.sessions,
        "train_stories": train_stories,
        "test_stories": test_stories,
        "model_type": "joint",
        "feature_spaces": ["acoustic", "speech", "language"],
        "mean_correlation": float(np.nanmean(joint_corrs)),
    }
    
    with open(os.path.join(joint_save_location, "metadata.json"), "w") as f:
        json.dump(metadata_joint, f, indent=2)

    # =========================================================================
    # 2. Individual Feature Space Models
    # =========================================================================
    individual_results = {}
    
    for space_name, space_indices in tqdm(
        feature_slices.items(),
        desc="Fitting individual feature space models",
        unit="space",
    ):
        print(f"\nFitting {space_name} individual model...")
        
        individual_save_location = os.path.join(save_location, "..", space_name, args.subject)
        os.makedirs(individual_save_location, exist_ok=True)
        
        space_kernelizer = ColumnKernelizer(
            [(space_name, Kernelizer(kernel="linear"), space_indices)],
            n_jobs=effective_n_jobs,
        )
        
        space_params = dict(
            alphas=alphas,
            progress_bar=True,
            n_iter=100,
            diagonalize_method="eigh",
            return_alphas=True,
        )
        
        space_model = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            space_kernelizer,
            KernelRidgeCV(kernels="precomputed", solver_params=space_params),
        )
        
        # 1. Fit & Predict using MASKED responses
        space_model.fit(delRstim, zRresp_masked)
        space_pred_backend = space_model.predict(delPstim)
        space_corrs_masked = to_numpy(
            correlation_score_split(zPresp_masked, space_pred_backend)
        )
        space_pred_masked = to_numpy(space_pred_backend)
        
        print(f"{space_name} model mean correlation: {np.nanmean(space_corrs_masked):.6f}")
        
        space_ridge = space_model[-1]
        
        # 2. Extract primal weights using StandardScaler-transformed features
        delRstim_scaled = to_numpy(space_model[0].transform(delRstim))
        space_feature_slice_scaled = delRstim_scaled[:, space_indices]

        try:
            space_weights = to_numpy(
                space_ridge.get_primal_coef([space_feature_slice_scaled])
            )
        except Exception:
            try:
                space_weights = to_numpy(
                    space_ridge.get_primal_coef(space_feature_slice_scaled)
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
            np.savez(os.path.join(individual_save_location, "alphas.npz"), alphas=space_ridge.best_alphas_)

        metadata_individual = {
            "subject": args.subject,
            "feature": args.feature,
            "modelname": args.modelname,
            "layer": int(selected_layer) if args.feature == "whisper" else None,
            "sessions": args.sessions,
            "train_stories": train_stories,
            "test_stories": test_stories,
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

    # Whisper options
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help="Existing HuthLab feature name or 'whisper'.",
    )
    parser.add_argument("--modelname", type=str, default="whisper-large")
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Manually selected Whisper decoder layer.",
    )

    # Feature preprocessing
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument(
        "--use-corr",
        action="store_true",
        help="Use correlation instead of signed variance explained.",
    )
    
    # Himalaya backend parameters
    parser.add_argument(
        "--backend",
        type=str,
        default="torch_cuda",
        choices=["numpy", "cupy", "torch_cuda"],
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
    run_subjects(args)