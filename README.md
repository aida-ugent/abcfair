# ABCFair: an Adaptable Benchmark approach for Comparing Fairness Methods

Code and results for the ABCFair benchmark. We evaluate on we evaluate 10 methods on 6 datasets 
(+ 1 from the unbiased labels in SchoolPerformance), 7 fairness notions, and 2 output formats, and 
3 sensitive feature formats. 

## Results
Putting all results here would lead to an obscenely large README, so
we provide two scripts to read out the benchmark results.

### 1: The Table Format
Results are presented in latex table code, with a row for each combination of sensitive feature format
and maximal fairness violation values k.

The table is generated with the ```generate_table_results.py``` script. 
To see the command line options, run ```python generate_table_results.py --help```. These include the dataset, 
the fairness notion with respect to which violation is measured, and the output format.

### 2: The Plot Format
Results are presented in an accuracy-fairness trade-off plot, for a range of fairness strengths. Each scatter point
is the mean test performance and fairness violation, with a confidence ellipse (using the standard error) around it.

The plot is generated with the ```generate_tradeoff_results.py``` script.
To see the command line options, run ```python generate_tradeoff_results.py --help```. These include the dataset,
the fairness notion with respect to which violation is measured, the output format, and the sensitive feature format.


## Running the Pipeline
To generate new results, the pipeline can be run with ```main.py```, which expects a config `.yaml` file as input. All 
logging is done using `wandb` (Weights and Biases), so you will need to login with your own account. Other required
packages are found in `requirements.txt`.

Larger benchmark experiments were done using the config/sweep_config `.yaml` files, following standard `wandb sweep` 
practice.