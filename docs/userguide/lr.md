# Logistic Regression

`gaia` supports logistic regression models for detecting ghost introgressed fragments. To view the subcommand information for logistic regression, users can use:

```
gaia lr -h
```

This will show information for four subcommands:

| Subcommand | Description |
| - | - |
| simulate | Simulate data for training |
| preprocess | Preprocess data for inference |
| train | Train a logistic regression model |
| infer | Infer ghost introgressed fragments with a given logistic regression model |

To see the arguments for each subcommand, for example, users can use:

```
gaia lr simulate -h
```

## `simulate`

Users can utilize the `simulate` subcommand to generate a training dataset by providing a demographic model in [DEMES](https://popsim-consortium.github.io/demes-docs/latest/introduction.html) format and a feature configuration file in YAML format.

### Demographic model

An example DEMES file is available [here](https://github.com/xin-huang/gaia/blob/main/examples/demog/ArchIE_3D19.yaml) and appears as follows:

```
doi:
    - https://doi.org/10.1371/journal.pgen.1008175
time_units: generations
demes:
    - name: Ancestral
      epochs:
          - {end_time: 2500, start_size: 10000}
    - name: Source
      ancestors: [Ancestral]
      start_time: 12000
      epochs: # https://github.com/sriramlab/ArchIE/blob/master/simulations/ms.sh#L7C48-L7C179
          - {end_time: 6120, start_size: 10000}
          - {end_time: 6000, start_size: 100}
          - {end_time: 0, start_size: 10000}
    - name: Reference
      ancestors: [Ancestral]
      epochs:
          - {end_time: 0, start_size: 10000}
    - name: Target
      ancestors: [Ancestral]
      epochs:
          - {end_time: 0, start_size: 10000}
pulses:
    - {sources: [Source], dest: Target, time: 2000, proportions: [0.02]}
```

This demographic model, based on the ms command found [here](https://github.com/sriramlab/ArchIE/blob/master/simulations/ms.sh#L7C48-L7C179), includes a population bottleneck in the Source population. It differs from the demographic model depicted in Figure 1 of [Durvasula and Sankararaman 2019](https://doi.org/10.1371/journal.pgen.1008175).

The visualization of this demographic model is illustrated below:

![ArchIE_3D19](https://github.com/xin-huang/gaia/blob/main/examples/demog/ArchIE_3D19.png?raw=true)

In this diagram, the dashed arrow indicates introgression from the Source population to the Target population, while the Reference population remains unaffected. For ghost introgression scenarios, genomes from the Source population are unavailable.

### Feature configuration

An example feature configuration file can be found [here](https://github.com/xin-huang/gaia/blob/main/examples/features/ArchIE.features.yaml) and is shown below:

```
Features:
    Ref distances:
        Minimum: true
    Tgt distances:
        All: true
        Mean: true
        Variance: true
        Skew: true
        Kurtosis: true
    Spectrum: true
    Private variant number: true
    Sstar:
        Genotype distance: 'ArchIE'
        Match bonus: 5000
        Max mismatch: 5
        Mismatch penalty: -10000
```

This file follows the [YAML](https://yaml.org/) format and begins with the keyword `Features`. For each input feature vector, six types of feautures can be estimated:

- Total variant number: Number of variants that exist in the reference and/or target population for a given sample. 
- Private variant number: Number of variants that only exist in the target population but not in the reference population for a given sample.
- Ref distances: Euclidean distances between the samples in the reference population and a given sample.
- Tgt distances: Euclidean distances between the samples in the target population and a given sample.
- Spectrum: The spectrum of mutation counts across all samples, based on the mutational sites within a given sample.
- Sstar: The S* statistic of a given sample based on the implementation in the [sstar](https://github.com/xin-huang/sstar) package.

For Ref and Tgt distances, further summary statistics need to be specified. These include:

- All: Distances between all samples in the specified population (Ref or Tgt) and a given sample.
- Minimum: Minimum of all distances.
- Maximum: Maximum of all distances.
- Mean: Mean of all distances.
- Median: Median of all distances.
- Variance: Variance of all distances.
- Skew: Skew of all distances.
- Kurtosis: Kurtosis of all distances.

For Sstar, four parameters need to be specified. They are:

- Genotype distance: Method to calculate genotype distance. Three methods are supported: `'ArchIE'` ([Durvasula and Sankararaman 2019](https://doi.org/10.1371/journal.pgen.1008175)), `'Vernot2014'` ([Vernot and Akey 2014](https://doi.org/10.1126/science.1245938)), and `'Vernot2016'` ([Vernot et al. 2016](https://doi.org/10.1126/science.aad9416)).
- Match bonus: Bonus for matching genotypes.
- Max mismatch: Maximum allowed mismatches.
- Mismatch penalty: Penalty for mismatches.

The above example feature configuration file specifies the same features used in the [ArchIE](https://github.com/sriramlab/ArchIE) implementation.

### Example

To create a training dataset with the above demographic model and feature configuration, use the following command:

```
gaia lr simulate --demes examples/demog/ArchIE_3D19.yaml --nref 5 --ntgt 5 --ref-id Reference --tgt-id Target --src-id Source --seq-len 50000 --phased --mut-rate 1.2e-8 --rec-rate 1e-8 --replicate 100 --output-prefix example.lr --output-dir examples/results/data/training --feature-config examples/features/ArchIE.features.yaml --nfeature 100 --nprocess 2 --seed 12345
```

In this command, we simulate a dataset with five diploid individuals from the Reference population and another five diploid individuals from the Target population, each with genomes of 50kb length (`--seq-len 50000`). The length of the simulated genomes in the training dataset should match the window length specified by the `--win-len` argument in the `preprocess` subcommand.

The `--replicate` argument specifies the number of replications per batch for the simulation. The simulation will continue until the number of feature vectors specified by the `--nfeature` argument is obtained.

The resulting training data can be found [here](https://github.com/xin-huang/gaia/blob/main/examples/results/data/training/example.lr.features) and looks like this:

```
Chromosome	Start	End	Sample	Sstar	Private_var_num	0_ton	1_ton	2_ton	3_ton	4_ton	5_ton	6_ton	7_ton	8_ton	9_ton	10_ton	Minimum_Ref_dist	Mean_Tgt_dist	Variance_Tgt_dist	Skew_Tgt_dist	Kurtosis_Tgt_dist	Tgt_dist_tsk_5_1	Tgt_dist_tsk_5_2	Tgt_dist_tsk_6_1	Tgt_dist_tsk_6_2	Tgt_dist_tsk_7_1	Tgt_dist_tsk_7_2	Tgt_dist_tsk_8_1	Tgt_dist_tsk_8_2	Tgt_dist_tsk_9_1	Tgt_dist_tsk_9_2	Label	Replicate
1	0	50000	tsk_5_1	0.0	3	0	3	1	0	1	2	1	3	0	4	3	2.449489742783178	3.48872744842987	1.7287807905720076	-1.6334122781798988	2.1947588783962457	0.0	3.0	3.0	3.605551275463989	3.605551275463989	3.7416573867739413	3.872983346207417	4.47213595499958	4.69041575982343	4.898979485566356	0	0
1	0	50000	tsk_5_2	19649.0	2	0	0	3	2	1	2	1	3	0	4	3	2.6457513110645907	3.254789503766206	2.9063452861733365	-1.1842207916814167	-0.16680552852599417	0.0	0.0	3.0	3.7416573867739413	3.7416573867739413	4.0	4.123105625617661	4.358898943540674	4.58257569495584	5.0	0	0
1	0	50000	tsk_6_1	0.0	4	0	3	4	1	1	1	1	0	0	4	3	3.7416573867739413	3.918161571996715	1.948009895728233	-2.0708242861935697	3.330551665081625	0.0	3.4641016151377544	3.7416573867739413	4.123105625617661	4.123105625617661	4.58257569495584	4.58257569495584	4.58257569495584	4.69041575982343	5.291502622129181	0	0
1	0	50000	tsk_6_2	-10000.0	8	0	6	4	1	0	1	1	0	0	4	3	4.242640687119285	4.16793656376827	2.128304800403537	-2.255462043427111	3.720880027655487	0.0	3.4641016151377544	4.47213595499958	4.58257569495584	4.58257569495584	4.795831523312719	4.795831523312719	4.795831523312719	4.898979485566356	5.291502622129181	0	0
1	0	50000	tsk_7_1	19649.0	2	0	0	3	2	1	2	1	3	0	4	3	2.6457513110645907	3.254789503766206	2.9063452861733365	-1.1842207916814167	-0.16680552852599417	0.0	0.0	3.0	3.7416573867739413	3.7416573867739413	4.0	4.123105625617661	4.358898943540674	4.58257569495584	5.0	0	0
1	0	50000	tsk_7_2	-10000.0	2	0	0	1	1	3	2	0	3	0	4	3	2.449489742783178	3.1104889980483437	2.8248581930202095	-1.022887702393171	-0.4271982278872448	0.0	0.0	2.449489742783178	3.605551275463989	3.605551275463989	3.7416573867739413	3.7416573867739413	4.58257569495584	4.58257569495584	4.795831523312719	0	0
1	0	50000	tsk_8_1	0.0	2	0	3	0	0	3	1	0	3	0	4	3	2.0	3.460592909120559	1.9242967173445067	-1.4179730355186297	1.1071104397774532	0.0	2.449489742783178	2.449489742783178	3.872983346207417	3.872983346207417	4.0	4.0	4.58257569495584	4.58257569495584	4.795831523312719	0	0
1	0	50000	tsk_8_2	0.0	5	0	8	1	2	0	1	1	3	0	1	3	3.872983346207417	4.321617155108232	2.2236251646742313	-2.329450209356624	4.126946818082276	0.0	4.358898943540674	4.358898943540674	4.58257569495584	4.58257569495584	4.58257569495584	4.69041575982343	5.291502622129181	5.291502622129181	5.477225575051661	0	0
1	0	50000	tsk_9_1	0.0	5	0	6	0	2	3	1	0	0	0	3	3	3.872983346207417	4.104968620314319	2.2492326262347495	-1.9035983826633205	2.72869867187561	0.0	3.605551275463989	3.605551275463989	3.872983346207417	4.69041575982343	4.898979485566356	4.898979485566356	5.0	5.0	5.477225575051661	0	0
1	0	50000	tsk_9_2	-10000.0	2	0	0	1	1	3	2	0	3	0	4	3	2.449489742783178	3.1104889980483437	2.8248581930202095	-1.022887702393171	-0.4271982278872448	0.0	0.0	2.449489742783178	3.605551275463989	3.605551275463989	3.7416573867739413	3.7416573867739413	4.58257569495584	4.58257569495584	4.795831523312719	0	0
```

The first line is the header, indicating the content of each column. For phased data, an additional number is appended to the sample name. For example, `tsk_5_1` denotes the first haploid genome from the sample `tsk_5`. The `X_ton` column contains the number of sites with `X` mutations. Following the [ArchIE](https://github.com/sriramlab/ArchIE) implementation, the `0_ton` column (i.e., non-segregating sites) contains only 0s. The `Tgt_dist_X` column represents the Euclidean distance between sample `X` and the sample in the current row. For example, the distance between `tsk_5_1` and `tsk_5_2` is `3`. In the `Label` column, `0`s represent non-introgressed fragments and `1`s denote introgressed fragments.

To determine whether a fragment is introgressed or not, the proportion of the length of the introgressed regions to the total length of a given squeuence is estimated.

By default:

- If the proportion >= 0.7, the fragment is considered as introgressed and labeled as `1`.
- If the proportion <= 0.3, the fragment is considered as non-introgressed and labeled as `0`.
- If 0.3 < the proportion < 0.7, the fragment is labeled as `-1` and discared in the training dataset.

Users can use the `--introgressed-prop` and `--non-introgressed-prop` arguments to change the proportions that determine whether a fragment is introgressed or not.

### Arguments

| Argument | Description |
| - | - |
| `--demes` | Demographic model in the DEMES format |
| `--nref` | Number of samples in the reference population |
| `--ntgt` | Number of samples in the target population |
| `--ref-id` | Name of the reference population in the demographic model |
| `--tgt-id` | Name of the target population in the demographic model |
| `--src-id` | Name of the source population in the demographic model |
| `--seq-len` | Length of the simulated genomes |
| `--ploidy` | Ploidy of the simulated genomes; default: 2 |
| `--phased` | Enable to use phased genotypes; default: False |
| `--mut-rate` | Mutation rate per base pair per generation for the simulation; default: 1e-8 |
| `--rec-rate` | Recombination rate per base pair per generation for the simulation; default: 1e-8 |
| `--replicate` | Number of replications per batch for the simulation, which will continue until the number of feature vectors specified by the `--nfeature` argument is obtained; default: 1 |
| `--output-prefix` | Prefix of the output file name |
| `--output-dir` | Directory of the output files |
| `--feature-config` | Name of the YAML file specifying what features should be used |
| `--nfeature` | Number of feature vectors should be generated; default: 1e6 |
| `--introgressed-prop` | Proportion that determines a fragment as introgressed; default: 0.7 |
| `--non-introgressed-prop` | Proportion that determinse a fragment as non-introgressed; default: 0.3 |
| `--keep-simulated-data` | Enable to keep simulated data; default: False |
| `--shuffle-data` | Enable to shuffle the feature vectors for training; default: False |
| `--force-balanced` | Enable to ensure a balanced distribution of introgressed and non-introgressed classes in the feature vectors for training; default: False |
| `--nprocess` | Number of processes for the simulation; default: 1 |
| `--seed` | Random seed for the simulation; default: None |

## `preprocess`

For real data, users should first use the `preprocess` subcommand to convert the data into feature vectors. Only [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf) files are accepted. Additionally, two files specifying the sample names in the reference and target populations are required. An example file can be found [here](https://github.com/xin-huang/gaia/blob/main/examples/results/data/test/example.lr.ref.ind.list) and is shown below:

```
tsk_0
tsk_1
tsk_2
tsk_3
tsk_4
```

In this file, each row records a sample name from the reference population.

Similar to the `simulate` subcommand, the `--feature-config` argument specifies a feature configuration as introduced in the `simulate` section above.

### Example

An example VCF file can be found [here](https://github.com/xin-huang/gaia/blob/main/examples/results/data/test/example.lr.vcf) and processed by the following command:

```
gaia lr preprocess --vcf examples/results/data/test/example.lr.vcf --chr-name 1 --ref examples/results/data/test/example.lr.ref.ind.list --tgt examples/results/data/test/example.lr.tgt.ind.list --feature-config examples/features/ArchIE.features.yaml --phased --output-prefix example --output-dir examples/results/data/test/ --win-len 50000 --win-step 10000 --nprocess 2
```

In this command, the genome is divided into 50 kb sliding windows with a step size of 10 kb. The output feature file is available [here](https://github.com/xin-huang/gaia/blob/main/examples/results/data/test/example.lr.features) and looks like:

```
Chromosome	Start	End	Sample	Sstar	Private_var_num	0_ton	1_ton	2_ton	3_ton	4_ton	5_ton	6_ton	7_ton	8_ton	9_ton	10_ton	Minimum_Ref_dist	Mean_Tgt_dist	Variance_Tgt_dist	Skew_Tgt_dist	Kurtosis_Tgt_dist	Tgt_dist_tsk_5_1	Tgt_dist_tsk_5_2	Tgt_dist_tsk_6_1	Tgt_dist_tsk_6_2	Tgt_dist_tsk_7_1	Tgt_dist_tsk_7_2	Tgt_dist_tsk_8_1	Tgt_dist_tsk_8_2	Tgt_dist_tsk_9_1	Tgt_dist_tsk_9_2
1	0	50000	tsk_5_1	0.0	1	0	0	0	7	13	6	3	4	1	1	1	4.123105625617661	5.309615536704651	10.207982852384578	-0.6658757575588138	-1.3441196355587541	0.0	1.0	1.0	4.69041575982343	6.855654600401044	7.615773105863909	7.874007874011811	7.874007874011811	8.06225774829855	8.12403840463596
1	0	50000	tsk_5_2	20140.0	8	0	2	4	3	17	4	6	3	1	1	1	5.0	5.85379453278392	7.133089567949097	-0.9046987815683206	-0.35118286017938694	0.0	3.4641016151377544	3.872983346207417	4.242640687119285	6.782329983125268	7.681145747868608	8.0	8.12403840463596	8.18535277187245	8.18535277187245
1	0	50000	tsk_6_1	0.0	1	0	0	0	7	13	5	3	4	1	1	1	4.0	5.218489072731758	11.167371797779243	-0.6825712521659232	-1.328941398563287	0.0	0.0	1.0	4.795831523312719	6.782329983125268	7.54983443527075	7.810249675906654	7.937253933193772	8.12403840463596	8.18535277187245
1	0	50000	tsk_6_2	8638.0	5	0	1	1	4	16	3	7	5	1	1	1	3.872983346207417	5.6313097608129095	6.288350377773256	-1.0075760379692977	-0.13277253797273136	0.0	3.1622776601683795	4.123105625617661	4.242640687119285	6.928203230275509	7.14142842854285	7.54983443527075	7.54983443527075	7.615773105863909	8.0
1	0	50000	tsk_7_1	0.0	4	0	9	3	3	0	5	9	3	0	0	1	5.744562646538029	6.259153831724486	4.622993310808681	-2.3907418064452797	4.226223568170759	0.0	5.744562646538029	6.708203932499369	6.782329983125268	6.782329983125268	6.855654600401044	7.14142842854285	7.280109889280518	7.615773105863909	7.681145747868608
1	0	50000	tsk_7_2	0.0	3	0	12	2	6	2	4	8	1	1	1	1	3.0	6.466955245681272	4.978489850355466	-2.3435056169064628	4.113562786769869	0.0	6.082762530298219	6.708203932499369	6.782329983125268	6.928203230275509	7.0710678118654755	7.3484692283495345	7.874007874011811	7.937253933193772	7.937253933193772
1	0	50000	tsk_8_1	0.0	1	0	0	0	7	13	5	3	4	1	1	1	4.0	5.218489072731758	11.167371797779243	-0.6825712521659232	-1.328941398563287	0.0	0.0	1.0	4.795831523312719	6.782329983125268	7.54983443527075	7.810249675906654	7.937253933193772	8.12403840463596	8.18535277187245
1	0	50000	tsk_8_2	20140.0	9	0	2	5	2	15	3	7	5	1	1	1	4.58257569495584	5.766933554670999	6.94247737600972	-0.9602798109927838	-0.34834860197407025	0.0	3.1622776601683795	3.4641016151377544	4.795831523312719	7.280109889280518	7.3484692283495345	7.810249675906654	7.810249675906654	7.874007874011811	8.12403840463596
1	0	50000	tsk_9_1	0.0	1	0	3	2	3	17	4	6	3	1	1	1	3.7416573867739413	5.873804462206417	6.498421139763988	-1.0258100060151762	0.12266501830016008	0.0	3.872983346207417	4.123105625617661	4.795831523312719	6.082762530298219	7.615773105863909	7.937253933193772	8.06225774829855	8.12403840463596	8.12403840463596
1	0	50000	tsk_9_2	0.0	1	0	1	3	3	14	6	2	3	0	1	1	5.0990195135927845	5.91590016026821	5.802125293738563	-1.2380448537795823	0.8920223089978263	0.0	4.69041575982343	4.795831523312719	4.795831523312719	5.744562646538029	7.0710678118654755	7.937253933193772	8.0	8.0	8.12403840463596
```

This output feature file resembles the one generated by the `simulate` subcommand but without the last two columns: the `Label` and `Replicate` columns.

### Arguments

| Argument | Description |
| - | - |
| `--vcf` | Name of the VCF file containing genotypes from samples |
| `--chr-name` | Name of the chromosome in the VCF file for being processed |
| `--ref` | Name of the file containing population information for samples without introgression |
| `--tgt` | Name of the file containing population information for samples for detecting ghost introgressed fragments |
| `--feature-config` | Name of the YAML file specifying what features should be used |
| `--phased` | Enable to use phased genotypes; default: False |
| `--ploidy` | Ploidy of genomes; default: 2 |
| `--output-prefix` | Prefix of the output files |
| `--output-dir` | Directory storing the output files |
| `--win-len` | Length of the window to calculate statistics as input features; default: 50000 |
| `--win-step` | Step size for moving windows along genomes when calculating statistics; default: 10000 |
| `--nprocess` | Number of processes for the training; default: 1 |

## `train`

Once the training data is created using the `simulate` subcommand, users can use the `train` subcommand to train a logsitic regression model.

### Example

An example command is shown below:

```
gaia lr train --training-data examples/results/data/training/examples.features --model-file examples/results/trained_model/example.lr.model --seed 12345
```

The trained model, which is a binary file, can be found [here](https://github.com/xin-huang/gaia/blob/main/examples/results/trained_model/example.lr.model).

### Arguments

| Argument | Description |
| - | - |
| `--training-data` | Name of the file containing features to training |
| `--model-file` | File storing the trained model |
| `--solver` | default: newton-cg |
| `--penalty` | default: None |
| `--max-iteration` | default: 10000 |
| `--seed` | Random seed for the training algorithm; default: None |
| `--scaled` | Enable to use scaled training data; default: False |

## `infer`

Once a trained model is obtained, users can utilize the `infer` subcommand to make predictions on the features processed by the `preprocess` subcommand.

### Example

An example command is shown below:

```
gaia lr infer --inference-data examples/results/data/test/example.lr.features --model-file examples/results/trained_model/example.lr.model --output-file examples/results/inference/example.lr.predictions
```

The predictions are available [here](https://github.com/xin-huang/gaia/blob/main/examples/results/inference/example.lr.predictions) and appears as follows:

```
Chromosome	Start	End	Sample	Non_Intro_Prob	Intro_Prob
1	0	50000	tsk_5_1	1.0	1.7830489487120413e-52
1	10000	60000	tsk_5_1	1.0	2.5749749680649507e-43
1	20000	70000	tsk_5_1	1.0	1.579513991055145e-39
1	30000	80000	tsk_5_1	1.0	2.1717524152864808e-52
1	40000	90000	tsk_5_1	1.0	4.148367079845042e-58
1	50000	100000	tsk_5_1	1.0	1.1664135728198555e-44
```

In this file, the last column represents the probability that a fragment is introgressed, and the second-to-last column represents the probability that a fragment is non-introgressed.

### Arguments

| Argument | Description |
| - | - |
| `--inference-data` | Name of the file storing features for inference |
| `--model-file` | Name of the file storing the trained model |
| `--output-file` | Name of the output file storing the predictions |
| `--scaled` | Enable to use scaled inference data; default: False |
