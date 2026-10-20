# ESMRank  
**A learn-to-rank approach for protein variant effect prediction**

ESMRank is a pipeline for ranking protein variants according to their predicted functional impact.  
The method frames variant effect prediction as a **learning-to-rank problem**, leveraging protein language models and sequence-derived features to prioritize mutations based on fitness, activity, or stability.

---

## Installation

### Requirements
- Python **3.10+**
- Linux / macOS
- A CUDA GPU (the ESM-2 feature step needs one)

### Install with pip

Install a `torch` build that matches your CUDA setup first (see https://pytorch.org), then:

```bash
pip install git+https://github.com/Riccarn/ESMRank.git@pip-package
```

This provides the `esmrank` command (and `python -m esmrank`).

### Download the pretrained models

The package does not bundle the models (~830 MB, hosted on Zenodo: https://zenodo.org/records/18773439). Fetch them once with:

```bash
esmrank download-models
```

They are stored in `~/.cache/esmrank/model` (or `--dest DIR`). At run time the model directory is looked up in this order: `--model-dir`, `$ESMRANK_MODEL_DIR`, `./model`, then the cache directory above.

### Install from source (development)

```bash
git clone https://github.com/Riccarn/ESMRank.git
cd ESMRank
git checkout pip-package
python -m venv ESMRank_venv && source ESMRank_venv/bin/activate
pip install -e ".[test]"
esmrank download-models
pytest
```

`requirements.txt` still lists the exact versions the pipeline was developed against; `pyproject.toml` declares compatible ranges.

---

## Usage

```bash
esmrank run --input INPUT [-csv] [-sub] [-indel] [-alanines]
            [--outdir DIR] [--model-dir DIR] [--gpus 0,1] [--batch-size N] [--esm-model NAME]
```

Results are written to `<outdir>/<input file name>/` (default `output/<name>/`): `predictions.csv`, plus `profile.tsv` and `profile.png` in `-sub` / `-alanines` mode.

The original entry point `run_esmrank_pipeline.sh [-h] --input INPUT ...` still works (it just calls `esmrank run`), so SLURM and Slivka setups keep working.

From Python:

```python
from esmrank import run_pipeline

run_pipeline("protein.fasta", sub=True, outdir="output")
```

---

## Input modes

ESMRank supports **two input modes**: CSV mode and FASTA mode.

---

### CSV mode

In CSV mode, the input file must be a CSV containing **three mandatory columns**:

| Column name | Description |
|------------|-------------|
| `hgvsp` | Variant identifier |
| `seq_wt` | Wild-type protein sequence |
| `seq_mu` | Mutant protein sequence |

Example:
```csv
hgvsp,seq_wt,seq_mu
p.A123V,MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQ...,MSEQNNTEMTFQIQRIYTKDISFEVPNAPHVFQ...
```

To run in CSV mode:
```bash
esmrank run --input variants.csv -csv
```

---

### FASTA mode

In FASTA mode, the input must be a **single wild-type protein sequence** in FASTA format.

The pipeline will automatically generate variants based on the selected mutation types:

- `-sub` : generate all possible **missense substitutions**
- `-indel` : generate all possible **insertions and deletions**
- `-alanines`: mutually exclusive to sub and indel, generate only alanines substitutions for each aminoacid

Example FASTA:
```fasta
>protein_X
MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQ...
```

Example execution:

generating all missenses variants:

```bash
esmrank run --input protein.fasta -sub
```
generating all single single indel

```bash
esmrank run --input protein.fasta -indel
```
generating all missenses and single indel

```bash
esmrank run --input protein.fasta -sub -indel
```
generate alanine scanning ( alanines substitution only ) 

```bash
esmrank run --input protein.fasta -alanines
```

---

## Command-line arguments

| Argument | Description |
|--------|-------------|
| `-h` | Show help message |
| `--input`, `-i` | Path to input file (CSV or FASTA) |
| `-csv` | Enable CSV input mode (default: FASTA mode) |
| `-sub` | Generate all possible missense substitutions |
| `-indel` | Generate all possible insertions and deletions |
| `-alanines` | Generate alanine substitutions for each position |
| `--outdir` | Output root directory (default: `output`) |
| `--model-dir` | Directory with the pretrained model files |
| `--gpus` | Comma-separated CUDA device ids (default: all) |
| `--batch-size` | Worker batch size for the ESM step (default: 128) |
| `--esm-model` | ESM-2 checkpoint name (default: `esm2_t33_650M_UR50D`) |

The environment variables `ESM_GPU_IDS`, `ESM_WORKER_BATCH_SIZE`, `ESM_MODEL_NAME` and `ESM_CHECKPOINT_DIR` are still honoured.

---

## Notes

- CSV and FASTA modes are **mutually exclusive**.
- In FASTA mode, at least one of `-sub`, `-indel` or `-alanines` must be specified.
- Runtime and memory usage scale with protein length and number of generated variants.

---

## Citation

If you use ESMRank in your research, please cite:

```bibtex
@article{esmrank,
  title={ESMRank reveals a transferable axis of protein mutational constraint from overlapping variant effect assays},
  author={Riccardo Arnese, Gennaro Gambardella},
  journal={TBD},
  year={TBD}
}
```
