# Generic and ML Workloads in an HPC Datacenter

This repository contains the scripts used to analyze the hardware and workload characteristics of the  Dutch datacenter SURF Lisa.
The main focus of this work is to compare generic and Machine Learning (ML) workloads in terms of hardware utilization, power consumption, and job characteristics.

## Setup

`sudo apt install cm-super texlive texlive-latex-extra texlive-fonts-recommended dvipng` (required for changing default font in matplotlib)

Set up a Python environment and install requirements to run the Jupyter Notebooks for (re-)producing results. In addition, Spark and Java are required for running the scripts.

`pip install -r requirements.txt`

In case of dependency errors, try running the requirements install command again, or manually install failing requirements.

Before running any script, download datasets, extract any zip files, and set variables for 4 paths to the datasets in `util/read_and_print_df.py` right at the top.

## Datasets

The datasets can be found at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11028933).

| Dataset Name                | Explanation               | Size      | Variable in Scripts           |
|-----------------------------|---------------------------|-----------|---------------------------|
| slurm_table_cleaned.parquet | Job data collected by SLURM | 31 MB | path_job_dataset           |
| prom_table_cleaned.parquet |  Node data collected by Prometheus | 16 GB | path_node_dataset           |
| prom_slurm_joined.parquet |  Joined Job and Node dataset      | 10 GB | path_job_node_joined_dataset |
| node_hardware_info.parquet |  Hardware configurations of each node | <1 MB| path_node_hardware_info      |

## Cluster Characterization Scripts

The scripts listed in the table below were used to produce the main results of the paper.
All scripts were tested on an AMD Epyc 64(128) core processor and with 1TB of RAM. Fewer cores/less RAM may also suffice, but runtimes will be higher.
The Python version used for scripts was 3.8.10, and Java JDK 11.0.22 for the Spark backend. Runtime per script for the described setup varies between 1-60 min (most time for script 4a). Script 3b shows our data integration process, however, it is not necessary to be run again (takes about 1h), as all our used datasets are already hosted processed and cleaned on Zenodo. The cumulative runtime for all scripts is 2-3 h.

| Script Name                                  | Section | Explanation                     |
|----------------------------------------------|----|----------------------------------------------------------------------------------------------------------------------------|
| 1_generic_vs_ml_overview.ipynb     | 1          | Figure 1, Table 5: Comparing generic to ML hardware/workload in the cluster.                                               |
| 2_node_and_rack_hardware_overview.ipynb  | 2     | Table 1, 2: Generate overview tables for cluster hardware stats.          |
| 2_workload_library_usage_overview.ipynb            | 2     | Workload overview through XALT logs.
| 3a_collected_node_data_example.ipynb  | 3a       | Data example visualization of collected raw Prometheus logs.       |
| 3b_data_integration.ipynb  | 3b       | Figure 2: Data integration process, combining job and node data.       |
| 4a_hardware_utilization_analysis.ipynb  | 4a      | Figure 3: Hardware utilization of nodes, distribution and boxplots.                                               |
| 4b_gpu_power_and_temperature_analysis.ipynb | 4b | Figure 4, Table 4: Hardware topology effects on GPU power vs temperature relations.                                            |
| 5a_job_arrivals_analysis.ipynb        | 5a       | Figure 5: Analysis if job submissions/arrivals over time.                                                                         |
| 5b_job_wait_and_run_time.ipynb      | 5b         | Figure 6: Job wait and runtime CDF plots.                                                                             |
| 5c_job_failures_analysis.ipynb      | 5c         | Figure 7: Failed jobs per hour bar plots.                                                                             |
| 5d_job_size_analysis.ipynb     | 5d              | Figure 8: Number of nodes and cores used per job.                                                                  |
| 6a_job_submissions_runtime_energy.ipynb  | 6a      | Figure 9: Job submissions, runtimes and energy consumption, grouped by state.                         |
| 6b_corr_job_states.ipynb     | 6b                 | Figure 10: Correlations of job termination states.                                                                    |
