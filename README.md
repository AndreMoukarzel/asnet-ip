# ASNets

This repository aims to reimplement the code used in the AAAI'18 paper
[Action Schema Networks: Generalised Policies with Deep Learning](https://arxiv.org/abs/1709.04271) 
and further described in the article
[ASNets: Deep Learning for Generalised Planning](https://arxiv.org/abs/1908.01362).

# MDP-IP

We also aim to make ASNets compatible with problems with imprecise probabilities,
consequentially needing to implement a teacher planner able to work on such
domain types.

For such, we aim to implement a planner such as described in
[Efficient solutions to factored MDPs with imprecise transition probabilities](https://www.sciencedirect.com/science/article/pii/S0004370211000026).

# Structure

This repository is structured as follows:

- `asnet/` contains our implementation of ASNets.
- `problems/` includes all problems that we used to train + test the network.
