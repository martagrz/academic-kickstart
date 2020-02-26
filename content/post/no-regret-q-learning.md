---
title: 'On Combining Q-Learning and No-Regret Learning'
date: 2020-02-26
markup: mmark
tags: 
- Q-Learning
- Multi-Agent 
- No-Regret Learning
draft: true
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
  
---

### Introduction

Value-based learning approaches used simultaneously in multi-agent environments lead to non-stationary and no convergence guarantees to the optimal policy. Instead, e


### No-Regret Learning 

Consider a Markov Decision Process (MDP) $M = \langle S, A, P, R, \gamma \rangle$, where $S$ is the state space, $A$ is the (finite) action space, $P : S \times A \rightarrow \Delta(S)$ is the transition probability, $R : S \times A \rightarrow \mathbb{R}$ is the (expected) reward function, assumed to be bounded, and $\gamma \in (0,1)$ is the discount rate. 

A regret-minimisation problem considers a single decision-maker playing a game against an adversary. Consider a set of $n$ actions $a_1, \dots, a_n$ and the following game setup: 
* At time $t=1,2,\dots,T$: 
	* A decision maker picks a mixed strategy $p^t$ (probability distribution over its actions $A$).
	* An adversary picks a cost vector $c^t : A \rightarrow [0, 1]$.
	* An action $a^t$ is chosen according to the distribution $p^t$ and the decision maker incurs a cost $$c^t(a^t)$$. The decision maker learns the entire cost vector $c^t$ not just the realised cost.  
	
We are not able to compare the cost of online decision-making to the cost of the best action sequence in hindsight - the benchmark of $$\sum^T_{t=1} \min_{a \in A}c^t(a)$$ is not achievable. Rather than comparing the expected cost of an algorithm to that of the best action *sequence* in hindsight, we compare it to the cost incurred by the best *fixed action* in hindsight. So, the benchmark becomes $$\min_{a\in A} \sum^T_{t=1} c^t(a)$$.  The time-averaged (external) regret of the action sequence $a_1, \dots, a_T$ with respect to the action $a$ is 

$$ \frac{1}{T} \left[ \sum_{t=1}^T c^t(a^t) - \sum_{i=1}^T c^t(a)  \right] $$

A no-regret algorithm is then defined as follows. Let $\mathcal{A}$ be an online decision-making algorithm. 
* An (adaptive) adversary for $\mathcal{A}$ is a function that takes as input the day $t$, the mixed strategies $p^1, \dots, p^t$ produced by $\mathcal{A}$ on the first $t$ days, and the realised actions $a^1, \dots,a^{t-1}$ of the first $t-1$ days, and produces as output a cost vector $c^t : [0,1] \rightarrow A$. 
* An online decision-making algorithm has no (external) regret if for every adversary for it, the expected regret wrt every action $a \in A$ is $o(1)$ as $T \rightarrow \infty$.    

### Counterfactual Regret Minimisation

Counterfactual regret minimisation (CFR) is a method to solve extensive form games of incomplete information. It works by using a no-regret algorithm to select actions. One copy of such an algorithm is used at each information set, which corresponds to a full history of play observed by a single agent. The resulting algorithm satisfies the global no-regret guarantee, so at least in two-player zero-sum games is guaranteed to converge to an optimal strategy through sufficient self-play. 

As noted in Kash et al (2019) CFR relies on two strong assumptions: perfect recall and a terminal state is eventually reached. 


### Local No-Regret Learning (LONR) 

As in CFR, LONR uses a copy of an arbitrary no-regret algorithm in each state. 



### Comments 

No-regret learning is an algorithm designed for non-stationary environments. 


### References
* Roughgarden T., CS364A: Algorithmic Game Theory Lecture #17: No-Regret Dynamics, [url](http://timroughgarden.org/f13/l/l17.pdf)
* Kash, I.A., Sullins, M. and Hofmann, K., 2019. Combining No-regret and Q-learning. *arXiv preprint arXiv:1910.03094.*
* Zinkevich, M., Johanson, M., Bowling, M. and Piccione, C., 2008. Regret minimization in games with incomplete information. In Advances in neural information processing systems (pp. 1729-1736).



