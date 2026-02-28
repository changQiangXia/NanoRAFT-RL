# NanoRAFT-RL Reproduction Error Log

Date: 2026-02-28
Environment: RTX 4090 24GB, Python 3.10.8, torch 2.1.2+cu121

## 1) Missing core dependencies after clone
- Symptom:
  - Most runtime imports failed (`transformers`, `peft`, `datasets`, `trl`, `langchain`, `llama_index`, etc.).
- Root cause:
  - Fresh environment only had partial packages preinstalled.
- Fix:
  - Installed project dependencies with `pip install -r requirements.txt`.
- Result:
  - Core scripts became importable.

## 2) `run_data_synthesis.py` import error: `No module named langchain_openai`
- Symptom:
  - `ModuleNotFoundError: No module named 'langchain_openai'`.
- Root cause:
  - `requirements.txt` missed `langchain-openai`, but code imports `from langchain_openai import ChatOpenAI`.
- Fix:
  - Added `langchain-openai==0.0.2` to `requirements.txt`.
  - Installed package.
- Result:
  - Data synthesis script passed CLI/import stage.

## 3) `sentence-transformers==2.2.2` incompatible with latest `huggingface_hub`
- Symptom:
  - `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`.
- Root cause:
  - `sentence-transformers==2.2.2` expects older `huggingface_hub` API.
- Fix:
  - Pinned `huggingface_hub==0.19.4` in `requirements.txt`.
  - Reinstalled compatible versions.
- Result:
  - `sentence_transformers` import recovered.

## 4) Network unreachable for Hugging Face model fetch
- Symptom:
  - Download attempts failed with `Network is unreachable`.
- Root cause:
  - Default network path unstable for `huggingface.co`.
- Fix:
  - Enabled accelerator command: `source /etc/network_turbo`.
  - Used `HF_ENDPOINT=https://hf-mirror.com`.
- Result:
  - Connectivity to `huggingface.co` and `hf-mirror.com` restored.

## 5) Security hardening for API key
- Symptom:
  - `configs/data_synthesis.yaml` contained plaintext Zhipu API key.
- Root cause:
  - Sensitive key was committed into config.
- Fix:
  - Cleared config value and switched to environment variable input (`ZHIPU_API_KEY`).
- Result:
  - Reproduction can continue without storing secrets in repo.

## 6) Interrupted index persist caused corrupted vector store file
- Symptom:
  - `JSONDecodeError` while loading `data/processed/index/default__vector_store.json`.
- Root cause:
  - The synthesis process was interrupted during index save, leaving a truncated JSON file.
- Fix:
  - Restored index files from `data.zip`:
    - `data/processed/index/default__vector_store.json`
    - `data/processed/index/docstore.json`
    - `data/processed/index/graph_store.json`
    - `data/processed/index/index_store.json`
- Result:
  - Index corruption removed; can proceed with existing prepared artifacts.

## 7) Long blocking time during embedding/index validation
- Symptom:
  - Validation stage took very long and looked stalled.
- Root cause:
  - First-time large embedding model download + heavy index loading.
- Fix:
  - Switched to practical reproduction path using provided prepared dataset (`data/synthetic`) for training/eval first, then optional full synthesis rerun.
- Result:
  - Keeps end-to-end reproduction moving without repeated long blocking startup.

## 8) SFT failed on Qwen2 due old transformers version
- Symptom:
  - `KeyError: 'qwen2'` while loading `Qwen/Qwen2-7B-Instruct`.
- Root cause:
  - `transformers==4.36.2` does not include Qwen2 config mapping.
- Fix:
  - Upgraded to `transformers==4.37.2`.
  - Updated `requirements.txt` accordingly.
- Result:
  - Qwen2 architecture loading path became compatible.

## 9) PPO startup redownload + slow/no-progress perception
- Symptom:
  - PPO stage appeared stalled for long periods.
  - Initially redownloaded full 7B model shards again.
- Root cause:
  - PPO script did not set cache dir like SFT script, so it missed existing local model cache.
  - Default PPO config (`total_steps=1000`) is too long for interactive reproduction.
  - Buffered stdout to file made step logs appear late.
- Fix:
  - Restarted PPO with explicit local cache env:
    - `HF_HOME=/root/autodl-tmp/NanoRAFT-RL/models/cache`
    - `HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/NanoRAFT-RL/models/cache`
  - Added a dedicated fast repro config `configs/ppo_rl_repro.yaml`:
    - reduced steps and generation length
    - denser logging for visible progress
  - Run with unbuffered Python (`python -u`) for real-time log flush.
- Result:
  - PPO pipeline remains complete but becomes tractable and observable.

## 10) Data synthesis smoke failed on mirror SSL handshake timeout
- Symptom:
  - `ProxyError` / SSL handshake timeout when fetching `all-MiniLM-L6-v2` from `hf-mirror.com`.
- Root cause:
  - Mirror endpoint intermittently unstable under current proxy route.
- Fix:
  - Kept `source /etc/network_turbo`, but switched to direct `huggingface.co` route (unset `HF_ENDPOINT`) for this step.
- Result:
  - Data synthesis smoke run succeeded (3/3 samples generated + distractor injection + split save).

## 11) PPO effective steps limited by dataloader length
- Symptom:
  - Repro PPO config `total_steps=60`, but run finished around step 34.
- Root cause:
  - Training loop iterates directly over `ppo_trainer.dataloader`; with 70 samples and batch size 2, one pass yields 35 batches.
- Fix:
  - Accepted one-pass PPO completion for this reproduction run (pipeline success + final model produced).
  - Note: to enforce strict `total_steps`, loop must cycle dataloader instead of single pass.
- Result:
  - PPO completed successfully with saved checkpoints and final adapter.

## 12) Evaluation script needed LoRA-adapter-aware loading
- Symptom:
  - SFT/PPO outputs are adapter directories, not standalone full base-model checkpoints.
- Root cause:
  - Original evaluator assumed `AutoModelForCausalLM.from_pretrained(model_path)` for all paths.
- Fix:
  - Updated `scripts/run_evaluation.py`:
    - detect `adapter_config.json`
    - load base model in 4-bit
    - attach adapter via `PeftModel.from_pretrained`
    - unify generation device handling
- Result:
  - Baseline/SFT/PPO three-model evaluation completed successfully.

---

Subsequent runtime issues (if any) will be appended below during full pipeline execution.
