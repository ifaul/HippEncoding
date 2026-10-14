import argparse
import json
import logging
import os
from os.path import join
from tqdm import tqdm

import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV, KernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import EM_DATA_DIR, REPO_DIR
from encoding_utils import apply_zscore_and_hrf, get_response
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from ridge_utils.utils import make_delayed
from whisper_feature_spaces import get_whisper_feature_space, get_decoder_layer_count


def configure_himalaya_backend(backend="numpy", n_jobs=-1, omp_threads=None):
    """
    Configure Himalaya backend for optimal CPU performance on HPC.
    """
    set_backend(backend)
    
    if omp_threads is None:
        omp_threads = os.cpu_count()
    
    # Set thread environment variables
    thread_env_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    
    for var in thread_env_vars:
        os.environ[var] = str(omp_threads)
    
    os.environ["OMP_NESTED"] = "FALSE"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    
    print(f"Himalaya backend: {backend}")
    print(f"OpenMP threads: {omp_threads}")
    print(f"Parallel jobs: {n_jobs if n_jobs > 0 else os.cpu_count()}")
    

def get_story_split(sessions):
    """
    Return ordered training and test story lists for the selected sessions.
    """
    with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)

    train_stories = []
    test_stories = []

    for session in sessions:
        session = str(session)

        if session not in sess_to_story:
            raise ValueError(f"Unknown session: {session}")

        stories, test_story = sess_to_story[session]

        train_stories.extend(stories)

        if test_story not in test_stories:
            test_stories.append(test_story)

    # Remove duplicates while preserving the dataset's order.
    train_stories = list(dict.fromkeys(train_stories))
    test_stories = list(dict.fromkeys(test_stories))

    overlap = set(train_stories).intersection(test_stories)
    if overlap:
        raise ValueError(
            "Training and test stories overlap: "
            f"{sorted(overlap)}"
        )

    return train_stories, test_stories


def select_optimal_decoder_layer(
    subjects,
    train_stories,
    modelname,
    candidate_layers,
    feature_dir=None,
    trim=5,
    ndelays=4,
    singcutoff=1e-10,
    use_corr=True,
):
    """
    Select optimal decoder layer using KernelRidgeCV.
    
    Evaluates candidate layers across multiple subjects in a leave-one-story-out
    validation scheme. Pools validation scores across subjects and selects the
    layer with the highest mean validation correlation.
    
    Parameters:
    -----------
    subjects : list of str
        Subject IDs to use for layer selection (typically first 3 subjects).
    train_stories : list of str
        Training stories for layer selection.
    modelname : str
        Name of Whisper model.
    candidate_layers : list of int
        Decoder layers to evaluate.
    feature_dir : str, optional
        Directory containing Whisper features.
    trim : int
        Number of TRs to trim from beginning/end of stories.
    ndelays : int
        Number of HRF delays.
    singcutoff : float
        Singular value cutoff for SVD.
    use_corr : bool
        Whether to use correlation (True) or variance explained (False).
    
    Returns:
    --------
    selected_layer : int
        Best decoder layer selected by 1-SE rule.
    layer_selection : dict
        Metadata about layer selection process.
    """
    if len(train_stories) < 2:
        raise ValueError(
            "At least two training stories are required for inner "
            "leave-one-story-out layer selection."
        )
    
    if not subjects:
        raise ValueError("At least one subject required for layer selection.")
    
    print(
        "Selecting decoder layer using "
        f"{len(subjects)} subjects and {len(train_stories)} training stories"
    )
    
    
    print(f"Candidate layers: {candidate_layers}")
    
    # Ridge alphas for KernelRidgeCV
    alphas = np.logspace(0, 3, 20)
    
    # Store validation scores per layer per subject
    layer_scores_by_subject = {layer: [] for layer in candidate_layers}
    
    # =========================================================================
    # Evaluate each subject
    # =========================================================================
    
    for subject in tqdm(
        subjects,
        desc="Selecting layer across subjects",
        unit="subject",
    ):
        print(f"\nProcessing subject: {subject}")
        
        # Load responses for this subject
        responses, valid_stories = get_response(train_stories, subject)
        
        if len(valid_stories) < 2:
            print(f"  WARNING: Only {len(valid_stories)} valid stories for {subject}, skipping")
            continue
        
        print(f"  Using {len(valid_stories)} valid training stories")
        
        for layer in candidate_layers:
            # Load feature space for this layer
            feature_space, available_stories = get_whisper_feature_space(
                valid_stories,
                modelname=modelname,
                layer=layer,
                feature_dir=feature_dir,
            )
            
            # Leave-one-story-out validation for this layer
            layer_validation_scores = []
            
            for validation_story in available_stories:
                inner_train_stories = [
                    s for s in available_stories if s != validation_story
                ]
                
                # Build delayed stimulus using make_delayed
                inner_train_stim_raw = apply_zscore_and_hrf(
                    inner_train_stories,
                    feature_space,
                    trim,
                    ndelays,
                )
                
                validation_stim_raw = apply_zscore_and_hrf(
                    [validation_story],
                    feature_space,
                    trim,
                    ndelays,
                )
                
                inner_train_resp, _ = get_response(
                    inner_train_stories,
                    subject,
                )
                validation_resp, _ = get_response(
                    [validation_story],
                    subject,
                )
                
                if inner_train_stim_raw.shape[0] != inner_train_resp.shape[0]:
                    raise ValueError(
                        f"Inner training mismatch for layer {layer}: "
                        f"stimulus has {inner_train_stim_raw.shape[0]} rows, "
                        f"response has {inner_train_resp.shape[0]} rows."
                    )
                
                if validation_stim_raw.shape[0] != validation_resp.shape[0]:
                    raise ValueError(
                        f"Validation mismatch for story {validation_story}: "
                        f"stimulus has {validation_stim_raw.shape[0]} rows, "
                        f"response has {validation_resp.shape[0]} rows."
                    )
                
                # Fit KernelRidgeCV model
                pipeline = make_pipeline(
                    StandardScaler(with_mean=True, with_std=False),
                    Kernelizer(kernel="linear"),
                )
                
                inner_train_kernel = pipeline.fit_transform(inner_train_stim_raw)
                validation_kernel = pipeline.transform(validation_stim_raw)
                
                model = KernelRidgeCV(alphas=alphas, cv=5)  # Think about whether there is any data leakage here due to 
                model.fit(inner_train_kernel, inner_train_resp)  # the autocorrelation of fMRI data
                
                validation_pred = model.predict(validation_kernel)
                
                # Compute correlation
                if use_corr:
                    validation_corr = correlation_score_split(
                        validation_resp,
                        validation_pred,
                    ).numpy(force=True)
                else:
                    # Variance explained
                    residual_var = np.var(validation_resp - validation_pred, axis=0)
                    resp_var = np.var(validation_resp, axis=0)
                    validation_corr = np.sqrt(
                        np.abs(1.0 - residual_var / resp_var)
                    ) * np.sign(1.0 - residual_var / resp_var)
                
                validation_score = float(np.nanmean(validation_corr))
                layer_validation_scores.append(validation_score)
            
            # Average validation score for this layer on this subject
            mean_layer_score = float(np.nanmean(layer_validation_scores))
            layer_scores_by_subject[layer].append(mean_layer_score)
            
            print(f"    Layer {layer}: {mean_layer_score:.6f}")
    
    # =========================================================================
    # Pool scores across subjects and select best layer
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("POOLED LAYER SELECTION RESULTS")
    print("=" * 80)
    
    means = {}
    standard_errors = {}
    
    for layer in candidate_layers:
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
            
            print(f"Layer {layer}: {mean_score:.6f} +/- {se:.6f}")
        else:
            print(f"Layer {layer}: SKIPPED (no valid scores)")
    
    if not means:
        raise ValueError("No valid layer scores obtained")
    
    best_layer = max(means, key=means.get)
    best_mean = means[best_layer]
    best_se = standard_errors[best_layer]
    
    # One-standard-error rule
    threshold = best_mean - best_se
    
    acceptable_layers = [
        layer for layer, score in means.items()
        if score >= threshold
    ]
    
    selected_layer = max(acceptable_layers)
    
    print(f"\nBest validation layer: {best_layer}")
    print(f"Selected layer by 1-SE rule: {selected_layer}")
    print("=" * 80)
    
    return selected_layer, {
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
        "selection_method": "leave-one-story-out with KernelRidgeCV",
        "ridge_method": "KernelRidgeCV",
        "use_corr": bool(use_corr),
    }

def run_encoding(args):
    # Configure backend
    configure_himalaya_backend(
        backend=args.backend,
        n_jobs=args.n_jobs,
        omp_threads=args.omp_threads,
    )
    
    if not args.sessions:
        raise ValueError("At least one session must be supplied.")

    if min(args.sessions) < 1 or max(args.sessions) > 5:
        raise ValueError("Sessions must satisfy 1 <= session <= 5.")

    if args.feature == "whisper":
        if args.optimal and args.layer is not None:
            raise ValueError(
                "Use either --layer or --optimal, not both."
            )

        if not args.optimal and args.layer is None:
            raise ValueError(
                "--feature whisper requires either --layer N "
                "or --optimal."
            )

    elif args.optimal:
        raise ValueError(
            "--optimal is only supported with --feature whisper."
        )

    train_stories, test_stories = get_story_split(args.sessions)
    all_stories = list(dict.fromkeys(train_stories + test_stories))

    selected_layer = args.layer
    layer_selection = None

    if args.feature == "whisper":
        if args.optimal:
            reference_story = train_stories[0]

            n_decoder_states = get_decoder_layer_count(
                reference_story,
                modelname=args.modelname,
                feature_dir=args.feature_dir,
            )

            optimal_stop = min(
                args.optimal_stop,
                n_decoder_states - 1,
            )

            if args.optimal_start > optimal_stop:
                raise ValueError(
                    f"No valid layers in range "
                    f"{args.optimal_start}..{args.optimal_stop}. "
                    f"The HDF5 file contains {n_decoder_states} "
                    "decoder states."
                )

            candidate_layers = list(
                range(args.optimal_start, optimal_stop + 1)
            )
            
            # Use first 3 subjects for layer selection
            calibration_subjects = ["UTS01", "UTS02", "UTS03"]
            
            selected_layer, layer_selection = (
                select_optimal_decoder_layer(
                    subjects=calibration_subjects,
                    train_stories=train_stories,
                    modelname=args.modelname,
                    candidate_layers=candidate_layers,
                    feature_dir=args.feature_dir,
                    trim=args.trim,
                    ndelays=args.ndelays,
                    singcutoff=args.singcutoff,
                    use_corr=args.use_corr,
                )
            )

            save_location = join(
                REPO_DIR,
                "results",
                args.feature,
                args.subject,
            )
            os.makedirs(save_location, exist_ok=True)
            
            with open(
                join(save_location, "layer_selection.json"),
                "w",
            ) as f:
                json.dump(layer_selection, f, indent=2)

        print(f"Using Whisper decoder layer: {selected_layer}")

        downsampled_feat, _ = get_whisper_feature_space(
            all_stories,
            modelname=args.modelname,
            layer=selected_layer,
            feature_dir=args.feature_dir,
        )

    else:
        if args.feature not in _FEATURE_CONFIG:
            available = " ".join(_FEATURE_CONFIG.keys())
            raise ValueError(
                f"Unknown feature '{args.feature}'. "
                f"Available features: {available}"
            )

        downsampled_feat = get_feature_space(
            args.feature,
            all_stories,
        )
    
    save_location = join(
        REPO_DIR,
        "results",
        args.feature,
        args.subject,
    )
    os.makedirs(save_location, exist_ok=True)
    
    
    print("Stimulus and response parameters:")
    print(f"trim: {args.trim}")
    print(f"ndelays: {args.ndelays}")

    delRstim = apply_zscore_and_hrf(
        train_stories,
        downsampled_feat,
        args.trim,
        args.ndelays,
    )
    print("delRstim:", delRstim.shape)

    delPstim = apply_zscore_and_hrf(
        test_stories,
        downsampled_feat,
        args.trim,
        args.ndelays,
    )
    print("delPstim:", delPstim.shape)

    zRresp, valid_train_stories = get_response(
        train_stories,
        args.subject,
    )
    print("zRresp:", zRresp.shape)

    zPresp, valid_test_stories = get_response(
        test_stories,
        args.subject,
    )
    print("zPresp:", zPresp.shape)

    train_stories = valid_train_stories
    test_stories = valid_test_stories

    if delRstim.shape[0] != zRresp.shape[0]:
        raise ValueError(
            "Training stimulus/response length mismatch: "
            f"{delRstim.shape[0]} vs {zRresp.shape[0]}"
        )

    if delPstim.shape[0] != zPresp.shape[0]:
        raise ValueError(
            "Test stimulus/response length mismatch: "
            f"{delPstim.shape[0]} vs {zPresp.shape[0]}"
        )

    # =========================================================================
    # MultipleKernelRidgeCV for joint model and individual feature spaces
    # =========================================================================
    
    print("\nFitting models with MultipleKernelRidgeCV...")
    
    # Get the feature slices from the concatenated feature matrix
    if args.feature == "whisper":
        n_acoustic = n_speech = n_language = 1280
    else:
        # For other features, infer from data
        n_acoustic = n_speech = n_language = downsampled_feat[train_stories[0]].shape[1] // 3
    
    # Account for delayed features: multiply by (1 + ndelays)
    n_features_total = delRstim.shape[1]
    delay_multiplier = n_features_total / (n_acoustic + n_speech + n_language)
    
    acoustic_slice = slice(
        0,
        int(n_acoustic * delay_multiplier),
    )
    speech_slice = slice(
        acoustic_slice.stop,
        acoustic_slice.stop + int(n_speech * delay_multiplier),
    )
    language_slice = slice(
        speech_slice.stop,
        speech_slice.stop + int(n_language * delay_multiplier),
    )
    
    feature_slices = {
        "acoustic": acoustic_slice,
        "speech": speech_slice,
        "language": language_slice,
    }
    
    print(f"Feature slices:")
    for name, s in feature_slices.items():
        print(f"  {name}: {s.start}..{s.stop} ({s.stop - s.start} features)")
    
    # Ridge alphas
    alphas = np.logspace(0, 3, 10)
    
    # =========================================================================
    # 1. Joint model (acoustic + speech + language)
    # =========================================================================
    
    print("\nFitting joint model (acoustic + speech + language)...")
    
    joint_save_location = os.path.join(
        save_location,
        "..",
        "joint",
        args.subject,
    )
    os.makedirs(joint_save_location, exist_ok=True)
    
    # Build pipeline for joint model
    joint_kernelizer = ColumnKernelizer(
        [("joint", Kernelizer(kernel="linear"), slice(None))],
        n_jobs=args.n_jobs if hasattr(args, 'n_jobs') else 1,
    )
    
    joint_params = dict(
        alphas=alphas,
        progress_bar=True,
        n_iter=100,
        diagonalize_method="svd",
    )
    
    joint_model = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        joint_kernelizer,
        MultipleKernelRidgeCV(kernels="precomputed", solver_params=joint_params),
    )
    
    joint_model.fit(delRstim, zRresp)
    joint_pred = joint_model.predict(delPstim)
    joint_corrs = correlation_score_split(zPresp, joint_pred).numpy(force=True)
    
    print(f"Joint model mean correlation: {np.nanmean(joint_corrs):.6f}")
    
    joint_ridge = joint_model[-1]
    joint_weights = joint_ridge.get_primal_coef()
    
    # Save joint model
    np.savez(os.path.join(joint_save_location, "weights"), joint_weights)
    np.savez(os.path.join(joint_save_location, "corrs"), joint_corrs)
    np.savez(os.path.join(joint_save_location, "predictions"), joint_pred)
    
    if hasattr(joint_ridge, "best_alphas_"):
        np.savez(os.path.join(joint_save_location, "alphas"), joint_ridge.best_alphas_)
    
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
    # 2. Individual feature space models
    # =========================================================================
    
    feature_space_names = ["acoustic", "speech", "language"]
    individual_results = {}
    
    for space_name, space_slice in tqdm(
        feature_slices.items(),
        desc="Fitting individual feature space models",
        unit="space",
    ):
        print(f"\nFitting {space_name} individual model...")
        
        individual_save_location = os.path.join(
            save_location,
            "..",
            space_name,
            args.subject,
        )
        os.makedirs(individual_save_location, exist_ok=True)
        
        # Build pipeline for this feature space
        space_kernelizer = ColumnKernelizer(
            [(space_name, Kernelizer(kernel="linear"), space_slice)],
            n_jobs=args.n_jobs if hasattr(args, 'n_jobs') else 1,
        )
        
        space_params = dict(
            alphas=alphas,
            progress_bar=True,
            n_iter=100,
            diagonalize_method="svd",
        )
        
        space_model = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            space_kernelizer,
            KernelRidgeCV(kernels="precomputed", solver_params=space_params),
        )
        
        space_model.fit(delRstim, zRresp)
        space_pred = space_model.predict(delPstim)
        space_corrs = correlation_score_split(zPresp, space_pred).numpy(force=True)
        
        print(f"{space_name} model mean correlation: {np.nanmean(space_corrs):.6f}")
        
        # Extract weights from the fitted model
        space_ridge = space_model[-1]
        space_weights = space_ridge.get_primal_coef()

        # Save individual model
        np.savez(os.path.join(individual_save_location, "weights"), space_weights)
        np.savez(os.path.join(individual_save_location, "corrs"), space_corrs)
        np.savez(os.path.join(individual_save_location, "predictions"), space_pred)

        # Also save selected alphas
        if hasattr(space_ridge, "best_alphas_"):
            np.savez(os.path.join(individual_save_location, "alphas"), space_ridge.best_alphas_)

        metadata_individual = {
            "subject": args.subject,
            "feature": args.feature,
            "modelname": args.modelname,
            "layer": int(selected_layer) if args.feature == "whisper" else None,
            "sessions": args.sessions,
            "train_stories": train_stories,
            "test_stories": test_stories,
            "model_type": space_name,
            "mean_correlation": float(np.nanmean(space_corrs)),
            "weights_saved": True,
        }

        with open(os.path.join(individual_save_location, "metadata.json"), "w") as f:
            json.dump(metadata_individual, f, indent=2)

        individual_results[space_name] = {
            "corrs": space_corrs,
            "predictions": space_pred,
            "weights": space_weights,
        }
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Joint model (all features): {np.nanmean(joint_corrs):.6f}")
    for space_name, results in individual_results.items():
        print(f"{space_name.capitalize():12s} model:        {np.nanmean(results['corrs']):.6f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset selection.
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help="Existing HuthLab feature name or 'whisper'.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )

    # Whisper options.
    parser.add_argument(
        "--modelname",
        type=str,
        default="whisper-large",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Manually selected Whisper decoder layer.",
    )
    parser.add_argument(
        "--optimal",
        action="store_true",
        help=(
            "Select the Whisper decoder layer using leave-one-story-out "
            "validation and bootstrap_ridge."
        ),
    )
    parser.add_argument(
        "--optimal-start",
        type=int,
        default=22,
    )
    parser.add_argument(
        "--optimal-stop",
        type=int,
        default=32,
    )

    # Feature preprocessing.
    parser.add_argument(
        "--trim",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ndelays",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--singcutoff",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "-use_corr",
        "--use-corr",
        action="store_true",
        help="Use correlation instead of signed variance explained.",
    )
    
    # Himalaya backend parameters.
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "cupy", "torch"],
        help="Himalaya backend: numpy (CPU) or cupy/torch (GPU).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Number of jobs for parallelization. "
            "-1 uses all CPUs, 1 disables parallelization."
        ),
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help=(
            "Number of OpenMP threads per job. "
            "If None, uses all available cores."
        ),
    )

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run_encoding(args)