<h2 align="center">The role of invariance in spectral complexity-based generalization bounds.</h2>

Deep convolutional neural networks (CNNs)
have been shown to be able to fit a random
labeling over data while still being able to
generalize well for normal labels. Describ-
ing CNN capacity through a posteriory mea-
sures of complexity has been recently pro-
posed to tackle this apparent paradox. These
complexity measures are usually validated by
showing that they correlate empirically with
GE; being empirically larger for networks
trained on random vs normal labels. Focus-
ing on the case of spectral complexity we in-
vestigate theoretically and empirically the in-
sensitivity of the complexity measure to in-
variances relevant to CNNs, and show several
limitations of spectral complexity that occur
as a result. For a specific formulation of spec-
tral complexity we show that it results in the
same upper bound complexity estimates for
convolutional and locally connected architec-
tures (which don’t have the same favorable
invariance properties). This is contrary to
common intuition and empirical results.

<h2> :boom: Files </h2>

```
GE_estimation_dense.py: Trains a dense neural network and estimates the network training and testing accuracy.

GE_estimation_locally_connected.py: Trains a locally connected neural network and estimates the network training and testing accuracy.

GE_estimation_convolutional.py: Trains a convolutional neural network and estimates the network training and testing accuracy.

utils_spectral_norm.py: Includes an implementation of Parseval regularization.

test_spectral_convolutional_variable_ab.py: Calculates the spectral norm for locally connected and convolutional 1d layers and computes empirical averages. Draws the theoretical average as well.
```

<h2> :memo: Citation </h2>

When citing this repository on your scientific publications please use the following **BibTeX** citation:

```bibtex
@article{pitas2019role,
  title={The role of invariance in spectral complexity-based generalization bounds},
  author={Pitas, Konstantinos and Loukas, Andreas and Davies, Mike and Vandergheynst, Pierre},
  journal={arXiv preprint arXiv:1905.09677},
  year={2019}
}

```

<h2> :envelope: Contact Information </h2>
You can contact me at any of my social network profiles:

- :briefcase: Linkedin: https://www.linkedin.com/in/konstantinos-pitas-lts2-epfl/
- :octocat: Github: https://github.com/konstantinos-p

Or via email at cwstas2007@hotmail.com

<h2> :warning: Disclaimer </h2>
This Python package has been made for research purposes.


