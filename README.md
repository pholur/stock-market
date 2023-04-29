# Scaling Law in Correlations of Returns in the S&P500

This repository contains code and data for the paper "Scaling Law in Correlations of Returns in the S&P500". In this paper, we investigate the emergent phenomena of the market-mode adjusted pairwise correlations of returns over different time scales and discover two scaling properties.

## Data

Raw data is of the format:
```
2005-11-01 00:00:00,21.39
2005-11-01 00:01:00,21.39
2005-11-01 00:02:00,21.39
2005-11-01 00:03:00,21.39
2005-11-01 00:04:00,21.39
2005-11-01 00:05:00,21.39
2005-11-01 00:06:00,21.39
```

The data used in this study is collected from the S&P500 market data over almost 20 years (2004-2020). The data is preprocessed and stored in the `Preprocessed` directory. The `Correlations` directory contains the market-mode adjusted pairwise correlations of returns over different time scales (τ) for each year. The `Histograms` directory contains the scaled and zero-shifted distributions of the c_i,j (τ)’s for each year.

## Code

All the code used for preprocessing, analysis, and plotting is stored in the `main` directory. The scripts are run in this order: `startup_indexer.py`, `correlation_indexer.py`, `histogram_indexer.py`. The `Plotters` folder provides a suite of plotting and visualization notebooks for the paper.

The data folder structure is as follows:
```
<Working Directory>
|
|-- Preprocessed
|----- data_{year}.npy
|
|-- Correlations
|----- corr_res_{year}_{tau}.pkl
|
|-- Histograms
|----- hist_{mode}_{year}_{tau}.pkl
```

## Paper

The paper can be accessed through this link: https://arxiv.org/pdf/2212.12703.pdf

## Citation

If you use any of the data or code presented in this repository, please cite the following paper:

```
@article{scaling_law_sp500,
  title={Scaling Law in Correlations of Returns in the S\&P500},
  author={Authors},
  journal={Journal},
  year={Year}
}
```