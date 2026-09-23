# Upstream software and citations

Cite T-REX using the code repository's `CITATION.cff`. Also credit the software,
models and data used in your work. The repositories below are official sources;
release-specific commits and model snapshots are recorded in `VERSIONS.json`
and `manifest.json` in the asset collection. A citation does not replace a license.

| Component | Official source | Research reference |
| --- | --- | --- |
| Proteina-Complexa | [NVIDIA-BioNeMo/Proteina-Complexa](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa) | Didi et al., *Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute*, ICLR 2026. [Paper](https://openreview.net/forum?id=qmCpJtFZra) |
| BindCraft | [martinpacesa/BindCraft](https://github.com/martinpacesa/BindCraft) | Pacesa et al., *One-shot design of functional protein binders with BindCraft*, Nature (2025). [DOI](https://doi.org/10.1038/s41586-025-09429-6) |
| BoltzGen | [HannesStark/boltzgen](https://github.com/HannesStark/boltzgen) | Stark et al., *BoltzGen: Toward Universal Binder Design* (2025). [DOI](https://doi.org/10.1101/2025.11.20.689494) |
| AlphaFold2 | [google-deepmind/alphafold](https://github.com/google-deepmind/alphafold) | Jumper et al., *Highly accurate protein structure prediction with AlphaFold*, Nature (2021). [DOI](https://doi.org/10.1038/s41586-021-03819-2) |
| AlphaFold-Multimer | [google-deepmind/alphafold](https://github.com/google-deepmind/alphafold) | Evans et al., *Protein complex prediction with AlphaFold-Multimer* (2021). [DOI](https://doi.org/10.1101/2021.10.04.463034) |
| ProteinMPNN | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Dauparas et al., *Robust deep learning-based protein sequence design using ProteinMPNN*, Science (2022). [DOI](https://doi.org/10.1126/science.add2187) |
| ColabDesign | [sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign) | Cite the software and the relevant AlphaFold/ProteinMPNN work above. |
| Foldseek | [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) | van Kempen et al., *Fast and accurate protein structure search with Foldseek*, Nature Biotechnology (2023). [DOI](https://doi.org/10.1038/s41587-023-01773-0) |
| MMseqs2 | [soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) | Steinegger and Söding, *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets*, Nature Biotechnology (2017). [DOI](https://doi.org/10.1038/nbt.3988); for clustering, *Clustering huge protein sequence sets in linear time*, Nature Communications (2018). [DOI](https://doi.org/10.1038/s41467-018-04964-5) |
| Qwen3.6-27B-FP8 | [QwenLM official GitHub organization](https://github.com/QwenLM); [exact model snapshot](https://huggingface.co/Qwen/Qwen3.6-27B-FP8/tree/ec4160bf26124fa57e6451d070ee0c459a36d5b7) | Qwen Team, *Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model* (2026). [Model release](https://qwen.ai/blog?id=qwen3.6-27b) |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP (2023). [Paper](https://arxiv.org/abs/2309.06180) |

The processed target structures are redistributed from the pinned public
Complexa data tree; individual source URLs and hashes are in the asset manifest,
and attribution is retained in `targets/ATTRIBUTION.txt`. Cite original structure
depositions where applicable. Checkpoint license notices remain beside each model.
