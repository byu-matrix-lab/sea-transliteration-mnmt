# SEA-Transliteration-MNMT

Codebase for *“When Scripts Diverge: Strengthening Low-Resource Neural Machine Translation Through Phonetic Cross-Lingual Transfer”*

---

## 1. Transliteration Scheme (Core Resource)

The file [`transliteration/all_transliterations.json`](transliteration/all_transliterations.json) defines the **complete transliteration mapping** used throughout this project.
It covers **Khmer, Lao, and Thai scripts**, mapping each character into three parallel representations:

| Field       | Description                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `ipa`       | International Phonetic Alphabet form                                                                                                       |
| `romanized` | Conventional Romanization                                                                                                                  |
| `cat`       | *Cognate-Aware Transliteration* (CAT) — a custom cross-lingual phonetic abstraction that aligns sound-equivalent characters across scripts |

This JSON serves as the **foundation for every transliteration and model input**.
Any model trained or evaluated with this repository relies on these mappings for script normalization and phonetic transfer.

---

## 2. Transliteration Utilities

### Script

`transliteration/transliterate.py`

### Purpose

Converts raw orthographic text (Khmer, Lao, Thai) into the desired transliteration form — IPA, romanized, or CAT — using the mappings in `all_transliterations.json`.

### Example Usage

```bash
python transliterate.py <json_path> <method> <text_file>
```

---

## 3. Model Training

The training code supports multiple **integration strategies** for transliteration in Multilingual NMT:

* **Shared-Encoder BART** (`train/shared_encoder_bart.py`)
* **Dual-Encoder BART** (`train/dual_encoder_bart.py`)
* **PyTorch Lightning Parent Model** (`train/model.py`)
* **Dataset and DataModule classes** (`train/dataset.py`)

### Tokenizer Training

Script: `train/train_tokenizer.py`
Trains ByteBPE Tokenizers.

```bash
python train_tokenizer.py <tokenizer_path> <vocab_size> <special_tokens> <text_file1> <text_file2> ...
```

### Main Training

Script: `train/train.py`
Configurable via YAML files (see `train/example_usage/example_config.yaml`).

Example:

```bash
python train.py <path_to_config>
```

Training configurations include:

* Model architecture (BART variant)
* Tokenizer path
* Learning rate, epochs, and batch sizes
* Data and checkpoint directories

---

## 4. Vocabulary Overlap and Metrics

### Purpose

Quantifies cross-lingual lexical overlap between transliteration schemes or corpora.

### Files

* `vocabulary_overlap/overlap_metrics.py` — main metric script
* `vocabulary_overlap/output.txt` — sample output

## ⚠️ Important Note

> **Bug fix notice:**
> After publication, a bug was discovered where PAD tokens were not excluded from the cross-entropy loss.
> This version **fixes that issue**, meaning your replication results may differ from the originally reported scores.

Dataset: [Paracrawl Bonus](https://paracrawl.eu/vbonus)

---

## Citation

If you use this repository or the transliteration mappings, please cite:

```
```

