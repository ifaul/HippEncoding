"""
Whisper feature spaces for HuthLab/deep-fMRI-dataset.

The narrative-enc Whisper files are expected at:

    <WHISPER_FEATURE_DIR>/<modelname>/narrative-<story>.pkl
    <WHISPER_FEATURE_DIR>/<modelname>/narrative-<story>.h5

The HDF5 datasets are:

    activations_conv : word x acoustic_dimension
    activations_enc  : word x encoder_dimension
    activations_dec  : word x decoder_layer x decoder_dimension

This module returns HuthLab-compatible feature spaces:

    {story: array(n_TRs, n_features)}

Decoder layer selection is handled externally by encoding.py.
"""

import os
import re
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from config import REPO_DIR
from ridge_utils.interpdata import lanczosinterp2D


WHISPER_FEATURE_DIR = os.environ.get(
    "WHISPER_FEATURE_DIR",
    os.path.join(REPO_DIR, "derivatives", "features"),
)

DEFAULT_MODEL = "whisper-large"


def normalize_word(word):
    """Normalize transcript tokens for conservative word matching."""
    if word is None:
        return ""

    word = str(word).lower().strip()
    word = word.replace("’", "'")
    word = word.replace("–", "-")
    word = word.replace("—", "-")

    # Retain apostrophes and hyphens because contractions may be meaningful.
    return re.sub(r"[^a-z0-9'-]", "", word)


def whisper_paths(story, modelname, feature_dir=None):
    if feature_dir is None:
        feature_dir = WHISPER_FEATURE_DIR

    root = os.path.join(feature_dir, modelname)
    pkl_path = os.path.join(root, f"narrative-{story}.pkl")
    h5_path = os.path.join(root, f"narrative-{story}.h5")

    return pkl_path, h5_path


def load_whisper_words(story, modelname=DEFAULT_MODEL, feature_dir=None):
    pkl_path, _ = whisper_paths(story, modelname, feature_dir)

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    df = pd.read_pickle(pkl_path).copy()

    required = {"word", "start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{pkl_path} is missing required columns: {sorted(missing)}"
        )

    df["word"] = df["word"].fillna("").astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")

    df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
    df["_normalized_word"] = df["word"].map(normalize_word)

    return df


def load_whisper_activations(
    story,
    activation,
    modelname=DEFAULT_MODEL,
    layer=None,
    feature_dir=None,
):
    """
    Load word-level Whisper activations.

    activation must be one of:
        conv, enc, dec
    """
    _, h5_path = whisper_paths(story, modelname, feature_dir)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    dataset_name = {
        "conv": "activations_conv",
        "enc": "activations_enc",
        "dec": "activations_dec",
    }[activation]

    with h5py.File(h5_path, "r") as h5:
        if dataset_name not in h5:
            raise KeyError(f"{dataset_name} not found in {h5_path}")

        states = np.asarray(h5[dataset_name][...], dtype=np.float32)

    if activation == "dec":
        if states.ndim != 3:
            raise ValueError(
                "Expected decoder activations with shape "
                "(words, layers, dimensions); got "
                f"{states.shape}"
            )

        if layer is None:
            raise ValueError("A decoder layer is required for decoder features.")

        if not 0 <= layer < states.shape[1]:
            raise ValueError(
                f"Decoder layer {layer} is outside [0, {states.shape[1] - 1}]"
            )

        states = states[:, layer, :]

    if states.ndim != 2:
        raise ValueError(
            f"Expected word-level activations with two dimensions; got {states.shape}"
        )

    return states


def nearest_word_index(df, target_word, target_time):
    """Find a matching word, falling back to temporal proximity."""
    target_word = normalize_word(target_word)

    matching = df[df["_normalized_word"] == target_word]
    if len(matching):
        distances = (matching["start"] - target_time).abs()
        return int(distances.idxmin())

    distances = (df["start"] - target_time).abs()
    return int(distances.idxmin())


def align_to_huth_words(df, states, huth_words, huth_times):
    """
    Align Whisper vectors to HuthLab's word sequence.

    Exact normalized word matches are preferred. Repeated words are resolved
    using timing. Unmatched words use nearest temporal matching.
    """
    if len(df) != states.shape[0]:
        raise ValueError(
            "Whisper transcript and activation lengths differ: "
            f"{len(df)} words versus {states.shape[0]} activation rows."
        )

    aligned = np.zeros(
        (len(huth_words), states.shape[1]),
        dtype=np.float32,
    )

    candidates = defaultdict(list)
    for index, word in enumerate(df["_normalized_word"]):
        if word:
            candidates[word].append(index)

    used = set()

    for row, (word, time) in enumerate(zip(huth_words, huth_times)):
        normalized = normalize_word(word)
        possible = candidates.get(normalized, [])

        if possible:
            available = [index for index in possible if index not in used]
            if not available:
                available = possible

            index = min(
                available,
                key=lambda i: abs(float(df.iloc[i]["start"]) - float(time)),
            )
        else:
            index = nearest_word_index(df, word, time)

        aligned[row] = states[index]
        used.add(index)

    return aligned


def get_story_feature(
    story,
    activation,
    modelname=DEFAULT_MODEL,
    layer=None,
    feature_dir=None,
):
    """
    Return one TR-aligned feature matrix for one story.
    """
    from feature_spaces import get_story_wordseqs
    
    wordseqs = get_story_wordseqs([story])
    wordseq = wordseqs[story]

    huth_words = np.asarray(wordseq.data, dtype=object)
    huth_word_times = np.asarray(wordseq.data_times, dtype=np.float32)
    huth_tr_times = np.asarray(wordseq.tr_times, dtype=np.float32)

    whisper_words = load_whisper_words(
        story,
        modelname=modelname,
        feature_dir=feature_dir,
    )
    states = load_whisper_activations(
        story,
        activation=activation,
        modelname=modelname,
        layer=layer,
        feature_dir=feature_dir,
    )

    word_features = align_to_huth_words(
        whisper_words,
        states,
        huth_words,
        huth_word_times,
    )

    tr_features = lanczosinterp2D(
        word_features,
        huth_word_times,
        huth_tr_times,
        window=3,
    )

    return np.asarray(tr_features, dtype=np.float32)


def get_whisper_feature_space(
    stories,
    modelname=DEFAULT_MODEL,
    layer=None,
    feature_dir=None,
    include_conv=True,
    include_enc=True,
):
    """
    Return the concatenated acoustic + encoder + decoder feature space.

    Decoder features use exactly one selected decoder layer.
    
    Skips any stories with missing Whisper files.
    Returns a tuple (feature_dict, valid_stories).
    """
    output = {}
    valid_stories = []

    for story in stories:
        try:
            pkl_path, h5_path = whisper_paths(
                story,
                modelname,
                feature_dir,
            )
            
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f"Whisper pickle not found: {pkl_path}"
                )
            
            if not os.path.exists(h5_path):
                raise FileNotFoundError(
                    f"Whisper HDF5 not found: {h5_path}"
                )
            
            blocks = []

            if include_conv:
                blocks.append(
                    get_story_feature(
                        story,
                        activation="conv",
                        modelname=modelname,
                        feature_dir=feature_dir,
                    )
                )

            if include_enc:
                blocks.append(
                    get_story_feature(
                        story,
                        activation="enc",
                        modelname=modelname,
                        feature_dir=feature_dir,
                    )
                )

            blocks.append(
                get_story_feature(
                    story,
                    activation="dec",
                    modelname=modelname,
                    layer=layer,
                    feature_dir=feature_dir,
                )
            )

            n_trs = {block.shape[0] for block in blocks}
            if len(n_trs) != 1:
                raise ValueError(
                    f"Feature blocks for {story} have incompatible TR counts: "
                    f"{sorted(n_trs)}"
                )

            output[story] = np.hstack(blocks).astype(np.float32)
            valid_stories.append(story)
            
        except (FileNotFoundError, OSError, KeyError) as e:
            print(
                f"WARNING: Whisper features unavailable for '{story}': "
                f"{type(e).__name__}"
            )
            print(f"  Details: {e}")
            print(f"  Skipping story '{story}'")
            continue

    if not output:
        raise ValueError(
            f"No valid Whisper features found for any of {stories}"
        )

    return output, valid_stories


def get_whisper_decoder_feature_space(
    stories,
    modelname=DEFAULT_MODEL,
    layer=None,
    feature_dir=None,
):
    """
    Return only one decoder layer as a feature space.
    """
    if layer is None:
        raise ValueError("layer must be provided")

    return {
        story: get_story_feature(
            story,
            activation="dec",
            modelname=modelname,
            layer=layer,
            feature_dir=feature_dir,
        )
        for story in stories
    }


def get_decoder_layer_count(
    story,
    modelname=DEFAULT_MODEL,
    feature_dir=None,
):
    """Return the number of decoder states saved in one HDF5 file."""
    _, h5_path = whisper_paths(story, modelname, feature_dir)

    with h5py.File(h5_path, "r") as h5:
        states = h5["activations_dec"]

        if states.ndim != 3:
            raise ValueError(f"Unexpected decoder shape: {states.shape}")

        return states.shape[1]