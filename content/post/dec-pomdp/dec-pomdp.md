---
title: 'Partial observability in multi-agent systems: Dec-POMDP'
date: 2020-02-26
markup: mmark
draft: true
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
  
---

### Introduction
When a single-agent is required to learn in a partially observable environment, we rely on a global belief state, meaning given the partially observable state $s$, we can infer the global environment based on a model. In a multi-agent setting, however, this is no longer a sufficient statistic since each agent has a local observation of their own state and no information about the states of the other agents. In general, this setting can be modelled by a decentralised POMDP (Dec-POMDP) (Oliehoek and Amato, 2016), which shares almost all elements such as the reward function and the transition model, as the multi-agent MDP (MMDP), except that each agent now only has its local observations of the system state $s$. With no accessibility to other agents’ observations, an individual agent cannot maintain a global belief state, the sufficient statistic for decision making in single-agent POMDPs.

Most algorithms applied in this setting assume co-operation between the agents, as opposed to competitive, zero-sum or mixed, general-sum games, and are based on a centralised-learning-decentralised-execution method. Centralisation means a central controller observes all the states, actions and rewards and makes action choices for each agent, whereas decentralisation means that each agent is autonomous and observes only their own states and rewards. This method first reformulates the problem as a centralised one using a simulator that generates the observation data of all agents. The policies are then optimised using this data and distributed to all agents for execution. 

### Method 

### Comments 

Dec-POMDPs have been shown to be NEXP-hard (Bernstein et al, 2002) and require super-exponential time to solve in the worst case. There have been multiple methods improving on this, including Monte Carlo sampling with policy iteration and the expectation-maximisation algorithm in Wu et al. (2010) and Wu et al. (2013). 

Extensions such as Dec-HDRQN (Omidshafrei et al), macro actions (Amato 2019) and MARL implemented at higher level (Amato 2014). Mac-CERTs (Xiao et al 2019) where macro actions concurrent with experience replay trajectories. 



### References
* Oliehoek and Amato, 2016
* Bernstein et al,  2002 
* Zhang, K., Yang, Z. and Başar, T., 2019. Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms. *arXiv preprint arXiv:1911.10635.*
* Wu et al 2010 
* Wu et al 2013



