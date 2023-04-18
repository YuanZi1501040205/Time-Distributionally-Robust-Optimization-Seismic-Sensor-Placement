# Similarity_Guided_CO2_Reduction_Catalyst_Discovery



This repo aims to combine explainable chemical compositions' similarity measurement(The Element Movers Distance-ElMD) technique and Greedy 
search to solve the CO2 reduction catalyst discovery problem. 

**UPDATE**
- 2022.03.02
    - Fine tuning for DRO>RO.

## Setup
pip install chama
conda install -c conda-forge glpk

!git clone https://github.com/YuanZi1501040205/Time-Distributionally-Robust-Optimization-Seismic-Sensor-Placement.git

Using Colab:
File->open notebook->GitHub->Similarity_Guided_CO2_Reduction_Catalyst_Discovery


Upload dataset: `CO2RR_predictions.pkl` (from literature's repo) or `CO_adsorp_energy_data.csv` (converted to csv)


Configure path to read dataset in read energy data session
```bash
pkl_path_GASpy_manuscript = './CO2RR_predictions.pkl'
```

runtime->run all

## Code References

[[1]](https://github.com/lrcfmd/ElMD) Element Movers Distance (ElMD)

[[2]](https://github.com/lrcfmd/ElM2D) A high performance mapping class to construct ElMD distance matrices from large datasets.

[[3]](https://github.com/ulissigroup/GASpy_manuscript) CO2 reduction catalyst DFT Dataset



## References
[[1]](https://www.nature.com/articles/s41929-018-0142-1) Tran, K., Ulissi, Z.W. Active learning across intermetallics to guide discovery of electrocatalysts for CO2 reduction and H2 evolution. (2018).

[[2]](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c03381) Hargreaves, Cameron J., et al. "The earth mover’s distance as a metric for the space of inorganic compositions." (2020).

[[2]](https://www.sciencedirect.com/science/article/pii/S2451929416301553) Davies, Daniel W., et al. "Computational screening of all stoichiometric inorganic materials." (2016).

