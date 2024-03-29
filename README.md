# ASNets [![Actions Status](https://github.com/AndreMoukarzel/asnet-ip/workflows/build/badge.svg)](https://github.com/AndreMoukarzel/asnet-ip/actions) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


This repository aims to reimplement the code used in the AAAI'18 paper
[Action Schema Networks: Generalised Policies with Deep Learning](https://arxiv.org/abs/1709.04271) 
and further described in the article
[ASNets: Deep Learning for Generalised Planning](https://arxiv.org/abs/1908.01362).

## MDP-IP

We also aim to make ASNets compatible with problems with imprecise probabilities,
consequentially needing to implement a teacher planner able to work on such
domain types.

For such, we aim to implement a planner such as described in
[Efficient solutions to factored MDPs with imprecise transition probabilities](https://www.sciencedirect.com/science/article/pii/S0004370211000026).

# Installation

You may install this repository as a package locally with the following command:
```Shell
pip install git+https://github.com/AndreMoukarzel/asnet-ip.git
```

If you have already cloned the repository, you may also install your local clone as a package with:
```Shell
cd asnet-ip
python install -e .
```

# Structure

This repository is structured as follows:

- `asnet/` contains our implementation of ASNets, and all auxiliary code used in their construction and training.
- `problems/` includes all problems that we used to train + test the network.


# Execution

If you are trying to use this package directly, it may be useful to generate problem instances as described [here](https://github.com/AndreMoukarzel/asnet-ip/blob/main/problems/PROBLEMS.md) 
in order to create the testing problems used by default by the scripts.

Some classes from the package can be executed by calling them directly. In this cases, instructions for each are as below:


### ASNet

Executing this class by itself will simply try to instantiate an ASNet for the given domain and problem instance. Can
be a good way to test if your domain or problem is correctly defined.

```Shell
cd asnet-ip
python -B -m asnet.asnet
```

Multiple arguments are available when running the asnet class:
- `--domain`/`-d`: Specify different problem domain.
- `--problem`/`-p`: Specify different problem instance.
- `--layer_num`/`-l`: Number of layers in the ASNet.
- `--image_name`/`-img`: Specify save path to save image representing the ASNet's structure.
- `--debug`: Turns on debug prints.
>Below is a demonstration of the usage of multiple arguments.

```Shell
cd asnet-ip
python -B -m asnet.asnet -d problems/deterministic_blocksworld/domain.pddl -p problems/deterministic_blocksworld/pb6.pddl --image_name asnet.jpg
```

### Trainer

Executing this class by itself will train an ASNet for the given domain in multiple problem instances. It takes a while.
The weights and biases of the network are saved to the `data/` folder after training.

```Shell
cd asnet-ip
python -B -m asnet.trainer
```
>Such as with the execution of the ASNet's class, optional arguments may be specified for personalization of the
execution.

**For details on available arguments use `--help` such as demonstrated below:**
```Shell
cd asnet-ip
python -B -m asnet.trainer --help
```


### Heuristics

Analogously, the heuristics can be also executed on specific problem instances by being referenced with the
`python -B -m` command, but with consideration of such files being inside the `heuristics/` subfolder.
Therefore, the heuristics can be run as follows:

```Shell
cd asnet-ip
python -B -m asnet.heuristics.lm_cut
python -B -m asnet.heuristics.hmax
```