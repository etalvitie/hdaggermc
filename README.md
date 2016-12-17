# hdaggermc
Source code related to the AAAI 2017 paper "Self-Correcting Models for Model-Based Reinforcement Learning" by Erik Talvitie. Please see the paper for descriptions of the algorithms and experiments.

Author: Erik Talvitie, 2016 (except where noted below)

Disclaimer/Apology: This is classic "research code." It was developed in a disorganized and ad-hoc manner as the needs of the project dictated, not via a disciplined design process. It was not developed to be general purpose, but rather to support specific experiments. As such, though I have tried to clean it up a little bit, it is very likely that it contains poor/puzzling design choices, vestigial appendages, and other oddities and/or flaws. Please forgive my mess. The main purpose of this release is to permit reproduction of the experiments and to archive the source code. If you would like to adapt some of this code for your own purposes, feel free to contact me with questions and I will do my best to help!

There are two main programs:

shooterDAggerUnrolled -- in this program, DAgger-MC and H-DAgger-MC use "unrolled" models, a separate model for each step in the rollout, responsible for predicting the observation at that step, given the output of the previous model.

shooterDAggerUndiscounted -- in this program, DAgger-MC and H-DAgger-MC use one model that is trained from data across all time steps in a rollout.

Requires:
boost (fairly light usage, could probably convert to C++11 without too much trouble)

To compile:
make all

To run:
Both programs take several command line arguments that parameterize the experiment and output. Run them with no arguments to see the help message.

Acknowledgements:
The models used by these programs are based on the Context Tree Switching (CTS) implementation developed by Joel Veness (http://jveness.info/software/default.html). The files cts.cpp/hpp, common.hpp, fastmath.cpp/hpp, icsilog.cpp/h, icsilogw.hpp, and jacoblog.hpp are from this implementation. I have also included Veness' readme file as cts-readme.txt.

Note that I have made minor alterations to cts.cpp/hpp, adding a few methods useful for this particular project.
