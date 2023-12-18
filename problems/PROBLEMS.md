# Problem Domains

This directory contains all problem domains used in experimentation of this
implementation of ASNets. The problem domains included are:
- BlockWorld (deterministic, probabilistic and with imprecise probabilities)
- Triangle Tireworld (probabilistic and with imprecise probabilities)
- Sys Admin (probabilistic and with imprecise probabilities)

The domains definitions and problem instance generators are found in their
respective subdirectories.

### Requirements

The problem instance generators require the [ippddl-parser](https://github.com/AndreMoukarzel/ippddl-parser)
package to be executed.

Be aware that generated problems' goals and initial states are defined randomly (following each domain's rules)
and therefore may differ when re-generated.