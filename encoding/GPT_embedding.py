import argparse
import json
import os
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2Model, GPT2TokenizerFast
from config import DATA_DIR

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
torch.set_num_interop_threads(2)

HFMODELS = {
    "gpt2-xl": "openai-community/gpt2-xl",
}


def main(
    stories: list[str],
    modelname: str,
    device: str,
    context_len: int = 1024,
    cache_dir: str | None = None,
):
    hfmodelname = HFMODELS[modelname]
    cache_dir = cache_dir or os.environ.get("HF_HOME")
    if cache_dir:
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.environ["HF_HOME"] = cache_dir

    print(f"Loading {hfmodelname} on {device} (Context Horizon W={context_len})...")
    tokenizer = GPT2TokenizerFast.from_pretrained(hfmodelname, cache_dir=cache_dir)
    model = GPT2Model.from_pretrained(
        hfmodelname,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
    )
    model = model.eval().to(device)

    # Save to a subfolder indicating context length (e.g., gpt2-xl/W1024)
    out_dir = os.path.join(DATA_DIR, "derivatives", "features", modelname, f"W_{context_len}")
    os.makedirs(out_dir, exist_ok=True)

    transcript_fpattern = os.path.join(DATA_DIR, "derivatives", "TextGrids", "{}_audio.json")

    for story in tqdm(stories, desc="Stories", ncols=80):
        transcript_file = transcript_fpattern.format(story)
        with open(transcript_file, "r") as f:
            d = json.load(f)
            df = pd.DataFrame(d["word_segments"])

        df["start"] = df["start"].ffill()
        df["end"] = df["end"].ffill()

        text_layer_embeddings = []

        with torch.no_grad():
            for idx in tqdm(range(len(df)), desc=f"Extracting {story}", leave=False, ncols=80):
                row = df.iloc[idx]

                # 1. Strictly causal history: slice preceding words up to current word idx
                # Cap history at roughly context_len words to keep string join fast
                hist_start_idx = max(0, idx - context_len)
                sub_words = df.iloc[hist_start_idx : idx + 1]["word"].astype(str).tolist()
                text = " " + " ".join(sub_words)

                # 2. Tokenize context string
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=False,
                    add_special_tokens=False,
                )
                input_ids = inputs["input_ids"]

                # 3. Truncate context to exactly context_len tokens if exceeded
                if input_ids.shape[-1] > context_len:
                    input_ids = input_ids[:, -context_len:]

                # 4. Target word token length (with leading space)
                target_tokens = tokenizer.tokenize(" " + str(row.word))
                target_len = max(1, len(target_tokens))

                # 5. Correct slice: target word occupies the trailing tokens
                target_slice = slice(-target_len, None)

                outputs = model(
                    input_ids=input_ids.to(device),
                    output_hidden_states=True,
                )

                # Shape: (n_layers + 1, seq_len, hidden_dim)
                hidden_states = torch.stack(outputs.hidden_states)[:, 0, :, :]

                # Mean-pool over the subword pieces of the target word: (n_layers + 1, hidden_dim)
                word_layer_states = hidden_states[:, target_slice, :].mean(dim=1)

                text_layer_embeddings.append(word_layer_states.cpu().numpy())

        # Save outputs
        text_layer_embeddings = np.stack(text_layer_embeddings)  # (N_words, 49, 1600)

        pkl_path = os.path.join(out_dir, f"{story}.pkl")
        h5_path = os.path.join(out_dir, f"{story}.h5")

        df.to_pickle(pkl_path)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(name="activations_enc", data=text_layer_embeddings)

        print(f"Saved {story}: shape {text_layer_embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelname", default="gpt2-xl", choices=["gpt2-xl"])
    parser.add_argument("-s", "--stories", type=str, nargs="+", required=True)
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
        help="Max context window in tokens (e.g., 1, 4, 16, 64, 256, 1024)",
    )
    parser.add_argument(
    "--device",
    default="mps" if torch.backends.mps.is_available() else "cpu",
    help="Device to run on (e.g., 'mps', 'cuda', 'cpu')",
)
    parser.add_argument("--cache-dir", default=None)

    args = parser.parse_args()
    main(
        stories=args.stories,
        modelname=args.modelname,
        device=args.device,
        context_len=args.context_len,
        cache_dir=args.cache_dir,
    )