# GENA-PsychNoise
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A novel computational pipeline integrating genetic noise analysis with GENA-LM foundation models for schizophrenia risk prediction from whole genome sequencing data.

## 🧬 Overview

This repository implements a groundbreaking approach to psychiatric genomics by combining:
- **Genetic noise quantification** - Novel metrics for measuring genomic instability and mutational burden
- **GENA-LM transformer models** - State-of-the-art genomic foundation models for sequence analysis
- **Integrated machine learning** - Multi-modal classification of schizophrenia risk

Our approach represents the first application of genomic language models to psychiatric genetic noise, potentially revealing new mechanisms underlying schizophrenia pathogenesis.

## 🎯 Key Features

- **Multi-scale genetic noise analysis** across individuals and genomic regions
- **Deep sequence embeddings** using pre-trained GENA-LM models
- **Comprehensive GWAS pipeline** with modern statistical approaches
- **Reproducible workflow** with containerized environments
- **Interactive visualizations** for exploratory data analysis
- **Modular design** allowing adaptation to other psychiatric disorders

## 📊 Dataset

- **Samples:** 100 schizophrenia patients + 100 healthy controls
- **Data types:** Whole genome sequencing (FASTQ, BAM, VCF formats)
- **Coverage:** ~30x average depth
- **Variants:** ~4-5M SNPs/indels per individual
- **Population:** Mixed ancestry with population stratification analysis

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for GENA-LM)
- ~50GB free disk space
- 16GB+ RAM

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SchizophreniaGenomicNoise-GENA.git
cd SchizophreniaGenomicNoise-GENA

# Create conda environment
conda env create -f environment.yml
conda activate genomic-noise

# Install additional dependencies
pip install -r requirements.txt

# Download GENA-LM models
python scripts/download_models.py
```

## 🚀 Quick Start

### 1. Data Preparation

```bash
# Prepare your VCF files
python scripts/prepare_data.py \
    --vcf data/raw/samples.vcf.gz \
    --phenotype data/phenotypes.txt \
    --output data/processed/

# Quality control
python scripts/quality_control.py \
    --input data/processed/filtered.vcf.gz \
    --plots results/qc_plots/
```

### 2. Genetic Noise Analysis

```bash
# Calculate noise metrics
python scripts/genetic_noise.py \
    --vcf data/processed/filtered.vcf.gz \
    --reference data/reference/GRCh38.fa \
    --output results/noise_metrics.csv

# Statistical comparison
python scripts/compare_noise.py \
    --metrics results/noise_metrics.csv \
    --phenotype data/phenotypes.txt
```

### 3. GENA-LM Analysis

```bash
# Extract sequences around variants
python scripts/extract_sequences.py \
    --vcf data/processed/filtered.vcf.gz \
    --reference data/reference/GRCh38.fa \
    --genes data/target_genes.bed \
    --window 10000

# Generate embeddings
python scripts/gena_embeddings.py \
    --sequences data/sequences/ \
    --model AIRI-Institute/gena-lm-bert-base-t2t \
    --output results/embeddings.npz
```

### 4. Integrated Analysis

```bash
# Combined machine learning model
python scripts/integrated_analysis.py \
    --noise results/noise_metrics.csv \
    --embeddings results/embeddings.npz \
    --phenotype data/phenotypes.txt \
    --output results/final_model/
```

## 📁 Repository Structure

```
SchizophreniaGenomicNoise-GENA/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── Dockerfile
├── config/
│   ├── analysis_config.yaml
│   └── model_config.yaml
├── scripts/
│   ├── prepare_data.py
│   ├── quality_control.py
│   ├── genetic_noise.py
│   ├── extract_sequences.py
│   ├── gena_embeddings.py
│   ├── integrated_analysis.py
│   └── utils/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_noise_visualization.ipynb
│   ├── 03_embedding_analysis.ipynb
│   └── 04_results_interpretation.ipynb
├── data/
│   ├── raw/           # Original FASTQ/BAM/VCF files
│   ├── processed/     # Filtered and cleaned data
│   ├── reference/     # Reference genome and annotations
│   └── sequences/     # Extracted sequences for GENA-LM
├── results/
│   ├── qc_plots/
│   ├── noise_analysis/
│   ├── embeddings/
│   └── final_model/
├── figures/           # Publication-ready figures
├── docs/             # Documentation
└── tests/            # Unit tests
```

## 🔬 Methodology

### Genetic Noise Metrics

Our pipeline quantifies multiple aspects of genetic noise:

1. **Mutational Load**
   - Total variant burden per individual
   - Rare variant enrichment (MAF < 0.01)
   - Functional variant ratios

2. **Genomic Instability**
   - Mutation spectrum signatures
   - Trinucleotide context patterns
   - Variant clustering indices

3. **Evolutionary Constraint**
   - Variants in conserved regions
   - Selection coefficient estimates
   - Constraint relaxation scores

### GENA-LM Integration

- **Sequence Extraction:** 10kb windows around variants in target genes
- **Model:** Pre-trained GENA-LM BERT transformer (4.5k-36k bp context)
- **Embeddings:** 768-dimensional sequence representations
- **Analysis:** Cosine similarity, clustering, dimensionality reduction

### Statistical Framework

- **Case-control comparisons:** Mann-Whitney U tests, effect sizes
- **Machine learning:** Random Forest, SVM, neural networks
- **Validation:** 5-fold cross-validation, permutation tests
- **Multiple testing:** Benjamini-Hochberg FDR correction

## 📈 Expected Results

Based on our preliminary analysis, we anticipate:

- **Genetic noise differences** between cases and controls (effect size: 0.3-0.5)
- **GENA-LM discrimination** of pathogenic vs benign variants (AUC > 0.75)
- **Combined model accuracy** for case classification (AUC > 0.80)
- **Novel risk genes** identified through sequence similarity


## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black scripts/
flake8 scripts/
```

## 📋 Requirements

### Computational Resources

- **CPU:** 8+ cores recommended
- **RAM:** 32GB+ for large datasets
- **GPU:** NVIDIA GPU with 8GB+ VRAM for GENA-LM
- **Storage:** 100GB+ free space

### Software Dependencies

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- NumPy, Pandas, SciPy
- Scikit-learn
- BCFtools, SAMtools
- See `requirements.txt` for complete list

## 🔒 Data Privacy

This repository contains only analysis code. No patient data or genomic sequences are included. Users must provide their own datasets following appropriate ethical guidelines and institutional approvals.
 

## 🏆 Acknowledgments

- **GENA-LM Team** at AIRI Institute for the foundation models
- **Psychiatric Genomics Consortium** for methodological frameworks 
- **Study participants** who made this research possible

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer:** This software is for research purposes only and is not intended for clinical use. All analyses should be validated independently before drawing biological conclusions.
