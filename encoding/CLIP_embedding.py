import json
import os
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from util.path import Path

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
torch.set_num_interop_threads(2)

# Model mapping for CLIP models
HFMODELS = {
    "clip-base": "openai/clip-vit-base-patch32",
    # Comparable large-scale CLIP text encoder (12 layers, 768 dim, 768 proj_dim)
    "clip-large": "openai/clip-vit-large-patch14",
    # Huge CLIP variant (24 layers, 1024 dim, 1024 proj_dim)
    "clip-huge": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}


def main(stories: list[str], modelname: str, device: str, cache_dir: str | None = None):

    hfmodelname = HFMODELS[modelname]
    cache_dir = cache_dir or os.environ.get("HF_HOME")
    if cache_dir:
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.environ["HF_HOME"] = cache_dir
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        print(f"Using Hugging Face cache directory: {cache_dir}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = CLIPTokenizer.from_pretrained(hfmodelname, cache_dir=cache_dir)
    model = CLIPTextModelWithProjection.from_pretrained(
        hfmodelname,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
    )

    print(
        f"Model : {hfmodelname} ({modelname})"
        f"\nLayers (text): {model.config.num_hidden_layers}"
        f"\nHidden Dim: {model.config.hidden_size}"
        f"\nProjection Dim: {model.config.projection_dim}"
        f"\nDevice: {device}"
    )
    model = model.eval()
    model = model.to(device)

    out_dir = f"/lustre/project/bdeen/HippEncoding/derivatives/features/{modelname}"
    os.makedirs(out_dir, exist_ok=True)

    transcript_fpattern = "/lustre/project/bdeen/HippEncoding/derivatives/TextGrids/{}_audio.json"
    
    for story in tqdm(stories, desc="Stories", ncols=80):
        print(story)

        # Load story transcript timing
        transcript_file = transcript_fpattern.format(story)
        with open(transcript_file, "r") as f:
            d = json.load(f)
            df = pd.DataFrame(d["word_segments"])
        df["start"] = df["start"].ffill()
        df["end"] = df["end"].ffill()

        # Setup context window examples per word segment
        examples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building", ncols=80):
            end_s = row.end
            start_s = max(0, end_s - 30)
            sub_df = df[(df.start > start_s) & (df.start < end_s)]

            text = " " + " ".join(sub_df.word.tolist())
            
            # CLIP Tokenizer handles text context (max sequence length 77)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=77,
                padding=False,
            )

            # Determine token slice count for the target word
            target_tokens = tokenizer.tokenize(" " + str(row.word))
            n_tokens = max(1, len(target_tokens))

            examples.append(
                dict(
                    word=row.word,
                    duration=row.end - row.start,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    n_tokens=n_tokens,
                )
            )

        # Run forward pass through CLIP text encoder
        text_layer_embeddings = []
        text_projection_embeddings = []

        with torch.no_grad():
            for example in tqdm(examples, desc="Extracting", ncols=80):
                # Extract target word token slice index
                target_emb_slice = slice(-(example["n_tokens"] + 1), -1)

                outputs = model(
                    input_ids=example["input_ids"].to(device),
                    attention_mask=example["attention_mask"].to(device),
                    output_hidden_states=True,
                )

                # 1. All hidden states across text transformer layers: (n_layers, 1, seq_len, hidden_dim)
                hidden_states = torch.stack(outputs["hidden_states"])

                # 2. Extract and mean-pool target word state across sub-word tokens: (n_layers, hidden_dim)
                word_layer_states = hidden_states[:, 0, target_emb_slice].mean(1)

                # 3. Extract final projected CLIP embedding (shared text-vision space): (proj_dim,)
                text_embeds = outputs["text_embeds"][0]

                text_layer_embeddings.append(word_layer_states.numpy(force=True))
                text_projection_embeddings.append(text_embeds.numpy(force=True))

        pkl_path = os.path.join(out_dir, f"{story}.pkl")
        h5_path = os.path.join(out_dir, f"{story}.h5")
        df.to_pickle(pkl_path)

        # Stack activation arrays across story words
        text_layer_embeddings = np.stack(text_layer_embeddings)
        text_projection_embeddings = np.stack(text_projection_embeddings)

        # Save extracted features to HDF5 file
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(name="activations_enc", data=text_layer_embeddings)
            f.create_dataset(name="activations_proj", data=text_projection_embeddings)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--modelname", default="clip-large", choices=["clip-base", "clip-large", "clip-huge"])
    parser.add_argument(
        "-s", "--stories", type=str, nargs="+", help="One or more story names"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")

    main(**vars(parser.parse_args()))