from pathlib import Path
import shutil

# Master mapping of target file paths to unique identifying markers
MAPPINGS = [
    # --- Turn 1: Setup & Data ---
    ("configs/smoke.yaml", ["L: 32", "quantizer: \"vq\"", "d: 64"]),
    ("configs/tinystories_dev.yaml", ["L: 128", "V: 8192", "K: 12"]),
    (".gitignore", ["# Data, models, results", "data/"]),
    ("requirements.txt", ["pyyaml", "tokenizers", "datasets"]),
    ("pytest.ini", ["[pytest]", "testpaths = tests"]),
    ("jncm/__init__.py", ["jncm: JEPA-style"]),
    ("jncm/config.py", ["Configuration dataclasses"]),
    ("jncm/utils.py", ["Utilities: seeding"]),
    ("jncm/data.py", ["Memmap dataset"]),
    ("jncm/eval/__init__.py", ["Evaluation utilities: generative perplexity"]),
    ("jncm/eval/genppl.py", ["Generative perplexity (Gen PPL)"]),
    ("scripts/prep_tinystories.py", ["Prepares TinyStories dataset"]),
    ("scripts/fetch_models.py", ["gpt2-large locally"]),
    ("tests/test_config.py", ["Tests for configuration dataclasses"]),
    ("tests/test_utils.py", ["Tests for seeding, JSONL logger"]),
    ("tests/test_data_prep.py", ["Tests for dataset packing logic"]),
    ("tests/test_data.py", ["Tests for MemmapDataset"]),
    ("tests/test_genppl.py", ["Tests for unigram entropy"]),

    # --- Turn 2: Levels & Tokenizer ---
    ("scripts/eval_data_row.py", ["Protocol gate evaluation script"]),
    ("tests/test_eval_data_row.py", ["Tests for protocol gate evaluation"]),
    ("jncm/levels.py", ["Multiscale level schedule"]),
    ("tests/test_levels.py", ["Tests for level schedule calculation"]),
    ("jncm/tokenizer/vq.py", ["Vector Quantizer with straight-through"]),
    ("jncm/tokenizer/noisy.py", ["Continuous code quantizer bounded by power"]),
    ("jncm/tokenizer/trunk.py", ["Trunk embedding, reconstruction operators"]),
    ("jncm/tokenizer/__init__.py", ["Tokenizer package: multiscale residual"]),
    ("tests/test_tokenizer.py", ["Comprehensive tests for MultiscaleTokenizer"]),

    # --- Documentation ---
    ("docs/STATE.md", ["STATE — jncm"]),
    ("docs/QUESTIONS.md", ["QUESTIONS — jncm"]),
]

def organize():
    # Find all downloaded ai_studio_code files or .md files in the current directory
    downloaded = list(Path(".").glob("ai_studio_code*")) + list(Path(".").glob("*.md"))
    downloaded = [f for f in downloaded if f.is_file() and f.name != "organize_downloads.py"]
    
    print(f"Found {len(downloaded)} downloaded files to sort...\n")
    
    success_count = 0
    for file_path in downloaded:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
            
        matched_target = None
        for target_path, markers in MAPPINGS:
            if all(m in content for m in markers):
                matched_target = target_path
                break
        
        if matched_target:
            dest = Path(matched_target)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(file_path, dest)
            print(f"✅ {file_path.name}  --->  {dest}")
            success_count += 1
        else:
            print(f"❌ Could not match: {file_path.name}")

    print(f"\nSuccessfully organized {success_count} / {len(downloaded)} files!")

if __name__ == "__main__":
    organize()