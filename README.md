# www-dom-xss-tools

This respository contains scripts for parsing training data and models related to WWW 2021 paper: Towards a Lightweight, Hybrid Approach for Detecting DOM XSS Vulnerabilities with Machine Learning

The datasets used in this study, as well as files for pre-trained models, can also be found at [the following link](https://doi.org/10.1184/R1/13870256).

## Requirements

This project relies on Tensorflow 1.14. Additional C modules for parsing word_bag objects are required, they are provided in js-build/libword_bag_ops.so.

## Running

`word_bag/word_bag.tf` is the main script that handles all training, testing and evaluation of the models.  
As a minimal example, the scripts `train.sh` and `eval.sh` are provided for quickstart.

## Configuration

To load configuration options, use the scripts provided in `config.sh` and the `configs` directory.

### General configuration

- `c_module_dir` : Points to the C dependencies. Relative paths are okay, so in most cases, `js-build` is appropriate.

### Model definition

- `n_features` : The hashsize of the word bag hashing, used as the size of the input to the embedding layer. In our study, a hashsize of 2^18, or 262144 was used.
- `dnn_embedding_size` : The size of the first embedding layer. In our study, we used an embedding size of 64.
- `batch_size` : Batch size used for training or evaluation.
- `classifier_name` : `custom_dnn_classifier` for a DNN, `linear_classifier` for a linear model.
- `dnn_hidden_units` : A nested list of integers, where each added number adds a new layer of the given size. For a first layer of N, our convention was [N, N/2, N/4]

### Data types

When loading the data (we provide both GZIP and LZMA options), the same configuration is used, whether training or testing is occurring.

#### For GZIP inputs:
- `compression_type`: `GZIP`

#### For LZMA inputs:
- `compression_type`: `LZMA`
- `file_format`: `lines-cache`
