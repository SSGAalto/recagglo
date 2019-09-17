
# Recursive Agglomerative Clustering (RecAgglo) for categorical data

Code and simple experimental setup for the Recursive Agglomerative Clustering (RecAgglo) algorithm introduced in **Detecting organized eCommerce fraud using scalable categorical clustering** (Marchal S. and Szyller S., ACSAC 2019).
This algorithm is initially designed to cluster orders placed on eCommerce websites with the aim to group fraudulent orders together. RecAgglo generates a large number of small clusters and it is best suited for processing data represented by attributes having high cardinality.
A detailed presentation of the algorithm is provided in Section 3 (Marchal S. and Szyller S., ACSAC 2019). RecAgglo clustering performance (Impurity + computation time) is assessed and compared to state-of-the-art categorical clustering algorithms in Section 7. Its capabilities for fraud detection are assessed in Section 8 (Marchal S. and Szyller S., ACSAC 2019).

## Requirements and setup

Python 3 packages: numpy, scipy, pandas, argparse. Also, you need a C compiler and Cython to compile the cython code. The code was tested with Python versions 3.6 and 3.7.

`pip3 install numpy, scipy, pandas, argparse, Cython`

Alternatively, you can use the provided `requirements.txt` file:

`pip3 install -r requirements.txt`

Compile the cython package `asym_linkage.pyx` with this command:

`python3 compile_asym_linkage.py build_ext --inplace`

## Contents

### Code

This packages consists of 3 main files:
- `main.py` main body of the experimental setup
- `clustering.py` containing the implementation of RecAgglo
- `parsing.py` custom argument parsing for the provided experiments
- `compile_asym_linkage.py` and `asym_linkage.pyx` containing fast implementation and compilation of the asymmetrical linkage function

### Datasets

Additionally, `test_data` directory contains standard datasets used to assess categorical clustering algorithms (from UCI Repository of Machine Learning Databases - https://archive.ics.uci.edu/ml/datasets):
- car
- census
- mushroom
- nursery

## Running the code

To run the clustering algorithm with default parameters, you need to provide just the input (data) and output file name.
- input file must be a comma separated value (CSV) file with one element per line
- output file is also a comma separated value (CSV) file, same shape as input file with one additional column containing the cluster index of each element

To run:

`python3 main.py -i test_data/mushroom.data -o result-mushroom.csv --verbose`

If you want to use custom parameters, invoke help to get more info:

`python3 main.py -h`

```
usage: main.py [-h] --infile INFILE --outfile OUTFILE [--delta_a INT]
               [--delta_fc INT] [--d_max FLOAT] [--rho_mc FLOAT]
               [--rho_s FLOAT] [--skip_index] [--verbose]

RecAgglo clustering. To run: python3 main.py --infile XYZ --outfile XYZ.
Override default args as necessary.

optional arguments:
  -h, --help            show this help message and exit
  --delta_a INT         Threshold for cluster sampling (default = 1000). Must
                        be > 0.
  --delta_fc INT        Threshold for full clustering (default = 1). Must be >
                        0.
  --d_max FLOAT         Distance threshold to split clusters (default = 0.5).
                        Must be > 0.
  --rho_mc FLOAT        Divider of max cluster according to n samples,
                        max_clust = n/mclust (default = 6.0). Must be > 0.
  --rho_s FLOAT         Multiplier of sqrt(n) for sample size (default = 0.5).
                        Must be > 0.
  --skip_index          Skip first column if it is an index and not a feature.
  --verbose             Verbose printing.

Required input/output file arguments:
  --infile INFILE, -i INFILE
                        CSV file containing input data.
  --outfile OUTFILE, -o OUTFILE
                        CSV output file containing input data + cluster
                        number.
```
