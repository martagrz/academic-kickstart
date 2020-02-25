---
title: 'Note on partial observability in multi-agent systems: Dec-POMDP'
date: 2020-02-25
markup: mmark
tags: 
- Reinforcement Learning 
- Partial Observability 
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
  - \usepackage{amsmath}
  
---

### Introduction
When a single-agent is required to learn in a partially observable environment, we rely on a global belief state, meaning given the partially observable state $s$ we can infer the global environment based on a model. In a multi-agent setting, however, this is no longer a sufficient statistic since each agent has a local observation of their own state and no information about the states of the other agents. In general, this setting can be modelled by a decentralised POMDP (Dec-POMDP) (Oliehoek and Amato, 2016), which shares almost all elements such as the reward function and the transition model, as the multi-agent MDP (MMDP), except that each agent now only has its local observations of the system state $s$. With no accessibility to other agents’ observations, an individual agent cannot maintain a global belief state, the sufficient statistic for decision making in single-agent POMDPs.

Most algorithms applied in this setting assume co-operation between the agents, as opposed to competitive, zero-sum or mixed, general-sum games, and are based on a centralised-learning-decentralised-execution method. Centralisation means a central controller observes all the states, actions and rewards and makes action choices for each agent, whereas decentralisation means that each agent is autonomous and observes only their own states and rewards. This method first reformulates the problem as a centralised one using a simulator that generates the observation data of all agents. The policies are then optimised using this data and distributed to all agents for execution. 

### Definition
A decentralised partially observable MDP (Dec-POMDP) is a tuple $$\langle I, S, \{ A_i \}, P, \{ \Omega_i \}, O, R, h \rangle$$ where 
* $I$ is a finite set of agents indexed $1,\dots, n$ 
* $S$ is a finite set of states, with distinguished initial state $s_0$
* $$A_i$$ is a finite set of actions available to agent $i$, and $\overrightarrow{A} = \otimes_{i \in I} A_i$ is the set of joint actions
* $P : S \times \overrightarrow{A}  \rightarrow \Delta(S)$ is a Markovian transition function, where $P(s' | s, \overrightarrow{a})$ denotes the probability that after taking joint action $\overrightarrow{a} $ in state $s$ a transition to state $s'$ occurs
* $$\Omega_i$$ is a finite set of observations available to agent $i$, and $\overrightarrow{\Omega}  = \otimes_{i \in I} \Omega_i$ is the set of joint observations
* $O : \overrightarrow{A}  \times S \rightarrow \Delta(\overrightarrow{\Omega})$ is an observation function. $O(\overrightarrow{o}|\overrightarrow{a} , s')$ denotes the probability of observing joint observation $\overrightarrow{o}$ given that joint action $\overrightarrow{a}$ was taken and led to state $s'$ 
* $R : \overrightarrow{A} \times S \rightarrow \mathbb{R}$ is a reward function. $R(\overrightarrow{a}, s')$ denotes the reward obtained after joint action $\overrightarrow{a}$ was taken and a state transition to $s'$ occurred
* If the Dec-POMDP has a finite horizon, that horizon is represented by a positive integer $h$ 

### Comments 

Dec-POMDPs have been shown to be NEXP-hard (Bernstein et al, 2002) and require super-exponential time to solve in the worst case. There have been multiple methods improving on this, including Monte Carlo sampling with policy iteration and the expectation-maximisation algorithm in Wu et al. (2010) and Wu et al. (2013). 

[comment]: <> (Extensions such as Dec-HDRQN (Omidshafiei et al, 2017), macro actions (Amato 2019) and MARL implemented at higher level (Amato 2014). Mac-CERTs (Xiao et al 2019) where macro actions concurrent with experience replay trajectories.)



### References
* Spaan, M., Amato, C., Zilberstein, S., 2011, Decision Making in Multiagent Settings: Team Decision Making, *AAMAS11 Tutorial*, [url](http://users.isr.ist.utl.pt/~mtjspaan/tutorialDMMS/tutorialAAMAS11.pdf)
* Oliehoek, F.A. and Amato, C., 2016. A concise introduction to decentralized POMDPs (Vol. 1). *Springer International Publishing*.
* Bernstein, D.S., Givan, R., Immerman, N. and Zilberstein, S., 2002. The complexity of decentralized control of Markov decision processes. *Mathematics of operations research*, 27(4), pp.819-840.
* Zhang, K., Yang, Z. and Başar, T., 2019. Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms. *arXiv preprint arXiv:1911.10635.*
* Wu, F., Zilberstein, S. and Chen, X., 2010, May. Point-based policy generation for decentralized POMDPs. In *AAMAS* (pp. 1307-1314).
* Wu, F., Zilberstein, S. and Jennings, N.R., 2013, June. Monte-Carlo expectation maximization for decentralized POMDPs. In *Twenty-Third International Joint Conference on Artificial Intelligence*.
* Omidshafiei, S., Pazis, J., Amato, C., How, J.P. and Vian, J., 2017, August. Deep decentralized multi-task multi-agent reinforcement learning under partial observability. In *Proceedings of the 34th International Conference on Machine Learning-Volume 70* (pp. 2681-2690).




