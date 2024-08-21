

# t-SNE, Kernel Density Estimation, and Their Integration

This repository contains the integration of [novel Adaptive Kernel Density Estimation algorithm](https://github.com/AlmondGod/kde-eval)  (Anand Majmudar and Matthew Tsui advised by Professor Erik Waingarten) with tSNE. 

By integrating KDE into t-SNE, we aim to reduce the time complexity while maintaining the quality of the dimensionality reduction. This could potentially make t-SNE applicable to larger datasets.

However, results were not satisfactory (see [results](/results)) with adaptive random sampling KDE in terms of execution times (sometimes increased) due to the variance-based performance of random sampling.

Thus, we attempt to use Kexin Rong and Paris Simelakis' [Hashing-Based Estimation](https://github.com/kexinrong/rehashing/tree/master/demo) to accelerate tSNE.

## Introduction 
### t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a dimensionality reduction technique used for visualizing high-dimensional data in 2 or 3 dimensions.

### Summary:

1. **Data Preparation**:
   - Normalize and zero-center the data

2. **Compute Input Similarities**:
   - Calculate Gaussian kernel for high-dimensional space
   - Use perplexity to determine sigma (often via binary search)
   - For non-exact version, use ball trees or vantage point trees

3. **Symmetrize Input Similarities**

4. **Gradient Descent**:
   - Initialize low-dimensional representation
   - Compute gradient and update points iteratively

### Detailed Algorithm:

1. Choose between exact and approximate algorithm
   - If N - 1 < 3 * perplexity, cannot proceed
   - If theta is 0, use exact (1 theta is most approximate, least accurate)
2. Set learning parameters (momentum, final momentum, eta)
3. Normalize input data
   - Zero center data then cap within a multidimensional box (-1, 1)
4. Compute perplexities and symmetrize
   - Exact method:
     - Compute squared Euclidean distance matrix DD: O(N²)
     - Compute Gaussian kernel row by row: O(N²)
     - For each point:
       - Find beta that achieves desired perplexity
       - Compute entropy H
       - Adjust beta using binary search
     - Symmetrize: P = (P + P^T) / (2N)
   - Approximate method (ball tree):
     - Build ball tree: O(N)
     - Compute K nearest neighbors: O(NK)
     - Compute perplexities on K nearest neighbors
     - Symmetrize sparse representation
5. Initialize solution and "lie" about P values (multiply by 12)
6. Main training loop:
   - Compute gradient:
     - Exact: O(N²) operations
       - Compute Q matrix (low-dimensional probabilities)
       - Gradient: (P - (Q / ∑Q)) * Q * (low-dim distance of points)
     - Approximate:
       - Compute edge forces (attractive forces)
       - Compute non-edge forces (repulsive forces)
       - Combine forces to get final gradient
   - Update gains (adaptive learning rates)
   - Apply momentum: Y = Y + uY
     - uY = momentum * uY - gains * eta * dY
   - Zero-mean the solution
7. Finish after predefined number of iterations

## Using Adaptive Random Sampling KDE for Acceleration
[adaptshell_kde.cpp](adaptshell_kde.cpp), [tsne_acc2.cpp][tsne_acc2.cpp]

### Motivation:

The bottleneck in t-SNE is computing pairwise similarities, which is O(N²) for exact t-SNE. Kernel Density Estimation can potentially reduce this complexity.

### Integration Points:

1. **Perplexity Computation**: 
   - Replace the O(N²) loop for computing Gaussian kernels with KDE
   - Current loop (to be replaced):
     ```
     for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
     P[nN + n] = DBL_MIN;
     
     sum_P = DBL_MIN;
     for(int m = 0; m < N; m++) sum_P += P[nN + m];
     
     double H = 0.0;
     for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
     H = (H / sum_P) + log(sum_P);
     ```
   - Challenge: Adapting KDE for the kernel function x * exp(-x/σ)

2. **Gradient Computation**:
   - Use KDE to estimate forces between points in low-dimensional space

Results could be improved with HBE, thus see below.

## Using Hashing-Based Estimation (HBE) for Acceleration

HBE is used to estimate the average of a kernel function over a dataset for a given query point.

### Adaptive HBE Steps:

1. Set parameters:
   - Effective diameter r = √(log(1/τ))
   - I = ceiling(log(1/τ) / log(2))
2. For each level i:
   - μᵢ[i] = (1 - γ) * μᵢ[i-1] (μᵢ[0] = (1 - γ))
   - For exponential kernel:
     - kᵢ[i] = min(0.25 * diam², 0.5 * diam * √(π/2))
     - wᵢ[i] = 2 * √(π/2) * kᵢ[i]
   - For Gaussian kernel:
     - tᵢ[i] = √(log(1/μᵢ[i]))
     - kᵢ[i] = 3 * ⌈r * tᵢ[i]⌉
     - wᵢ[i] = kᵢ[i] / (tᵢ[i] * √(2/π))
   - Mᵢ[i] = exp(1.5) / (√μᵢ[i] * ε²)
3. Create sketch or uniform HBE levels
4. Evaluate query:
   - Find hash bucket for query
   - Estimate kernel average using points in the bucket

## Applying Hashing to t-SNE

### Integration Points:

1. **Perplexity Computation**: 
   - Replace the O(N²) loop for computing Gaussian kernels with HBE
   - Current loop (to be replaced):
     ```
     for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
     P[nN + n] = DBL_MIN;
     
     sum_P = DBL_MIN;
     for(int m = 0; m < N; m++) sum_P += P[nN + m];
     
     double H = 0.0;
     for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
     H = (H / sum_P) + log(sum_P);
     ```
   - Challenge: Adapting HBE for the kernel function x * exp(-x/σ)

2. **Gradient Computation**:
   - Use HBE to estimate forces between points in low-dimensional space

### Implementation Considerations for HBE-tSNE Integration:

1. Choose between Sketch or Uniform HBE
2. Modify kernel function in configuration
3. Adapt t-SNE to use HBE for estimating perplexity and gradient

### Challenges:

- Adapting HBE code for t-SNE's specific kernel function
- Balancing accuracy and speed in the approximation
- Ensuring the approximation doesn't negatively impact t-SNE's quality

By integrating HBE into t-SNE, we aim to reduce the time complexity while maintaining the quality of the dimensionality reduction. This could potentially make t-SNE applicable to larger datasets.


# Main Branch README
[![Build Status](https://travis-ci.org/lvdmaaten/bhtsne.svg)](https://travis-ci.org/lvdmaaten/bhtsne)

This software package contains a Barnes-Hut implementation of the t-SNE algorithm. The implementation is described in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf).


# Installation #

On Linux or OS X, compile the source using the following command:

```
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
```

The executable will be called `bh_tsne`.

On Windows using Visual C++, do the following in your command line:

- Find the `vcvars64.bat` file in your Visual C++ installation directory. This file may be named `vcvars64.bat` or something similar. For example:

```
  // Visual Studio 12
  "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat"

  // Visual Studio 2013 Express:
  C:\VisualStudioExp2013\VC\bin\x86_amd64\vcvarsx86_amd64.bat
```

- From `cmd.exe`, go to the directory containing that .bat file and run it.

- Go to `bhtsne` directory and run:

```
  nmake -f Makefile.win all
```

The executable will be called `windows\bh_tsne.exe`.

# Usage #

The code comes with wrappers for Matlab and Python. These wrappers write your data to a file called `data.dat`, run the `bh_tsne` binary, and read the result file `result.dat` that the binary produces. There are also external wrappers available for [Torch](https://github.com/clementfarabet/manifold), [R](https://github.com/jkrijthe/Rtsne), and [Julia](https://github.com/zhmz90/BHTsne.jl). Writing your own wrapper should be straightforward; please refer to one of the existing wrappers for the format of the data and result files.

Demonstration of usage in Matlab:

```matlab
filename = websave('mnist_train.mat', 'https://github.com/awni/cs224n-pa4/blob/master/Simple_tSNE/mnist_train.mat?raw=true');
load(filename);
numDims = 2; pcaDims = 50; perplexity = 50; theta = .5; alg = 'svd';
map = fast_tsne(digits', numDims, pcaDims, perplexity, theta, alg);
gscatter(map(:,1), map(:,2), labels');
```

Demonstration of usage in Python:

```python
import numpy as np
import bhtsne

data = np.loadtxt("mnist2500_X.txt", skiprows=1)

embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])
```

### Python Wrapper

Usage:

```bash
python bhtsne.py [-h] [-d NO_DIMS] [-p PERPLEXITY] [-t THETA]
                  [-r RANDSEED] [-n INITIAL_DIMS] [-v] [-i INPUT]
                  [-o OUTPUT] [--use_pca] [--no_pca] [-m MAX_ITER]
```

Below are the various options the wrapper program `bhtsne.py` expects:

- `-h, --help`                      show this help message and exit
- `-d NO_DIMS, --no_dims`           NO_DIMS
- `-p PERPLEXITY, --perplexity`     PERPLEXITY
- `-t THETA, --theta`               THETA
- `-r RANDSEED, --randseed`         RANDSEED
- `-n INITIAL_DIMS, --initial_dims` INITIAL_DIMS
- `-v, --verbose`
- `-i INPUT, --input`               INPUT: the input file, expects a TSV with the first row as the header.
- `-o OUTPUT, --output`             OUTPUT: A TSV file having each row as the `d` dimensional embedding.
- `--use_pca`
- `--no_pca`
- `-m MAX_ITER, --max_iter`         MAX_ITER


# Adaptive shell kernel density estimation
g++ -std=c++11 -O3 adaptshell_kde.cpp -o adaptshell_kde
./adaptshell_kde


# tsne_acc
g++ -std=c++11 sptree.cpp tsne_acc.cpp tsne_main.cpp -o bh_tsne -O2 
python3 tsne_test.py