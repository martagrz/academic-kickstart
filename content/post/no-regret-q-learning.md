---
title: 'On Combining Q-Learning and No-Regret Learning'
date: 2020-03-02
markup: mmark
tags: 
- Q-Learning
- Multi-Agent 
- No-Regret Learning
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
  
---

# Introduction

Policy-based reinforcement learning methods used in single-agent environments can be extended to the multi-agent setting, mainly through **self-play**, where the agent learns how to play a game by repeatedly playing itself. These methods aim to achieve optimal performance when other agents play arbitrarily by minimising the (external) regret, meaning that the mixed policy chosen has to be at least as good as any fixed policy available to the agent in hindsight. An algorithm with negligible average overall regret is called no-regret or Hannan-consistent. There are two main families of Hannan-consistent multi-agent reinforcement learning methods with a two-player zero-sum game: fictitious play and counterfactual regret minimisation. Here, we focus on the latter. 

## Counterfactual regret minimisation

Counterfactual regret minimisation (CFR) (Zinkevich et al, 2008) is a framework for solving extensive-form games with incomplete information which establishes explicit bounds on performance, which yield rates of convergence to the Nash equilibrium. It relies on self-play algorithms, implying that the game should be **symmetric** (both agents have the same set of actions, states and rewards). 

The methods under this framework rely on a no-regret algorithm to select actions. In a CFR framework, one copy of such an algorithm is used at each information set (set of states that contain the same information), which corresponds to a full history of play observed by a single agent. In a perfect information case, the information set is a singleton, whereas in the imperfect information set there is more than one element in the set and the agents cannot distinguish between histories of the same information set. In either case, the resulting algorithm satisfies the global no-regret guarantee, so at least in two-player zero-sum games it is guaranteed to converge to an optimal strategy through sufficient self-play. 

CFR relies on two strong assumptions: perfect recall, such that the agents remember exactly the states and actions chosen, and that a terminal state is eventually reached in each iteration. Additionally, a drawback of the vanilla CFR method is that the entire game tree needs to be traversed in each iteration, which is computationally expensive. For a thorough explanation on CFR see [this post](https://www.quora.com/What-is-an-intuitive-explanation-of-counterfactual-regret-minimization) by Michael Johanson, one of authors on the original paper. 

# Method

## Set-up

Consider a Markov Decision Process (MDP) $M = \langle S, A, P, R, \gamma \rangle$, where $S$ is the state space, $A$ is the (finite) action space, $P : S \times A \rightarrow \Delta(S)$ is the transition probability, $R : S \times A \rightarrow \mathbb{R}$ is the (expected) reward function, assumed to be bounded, and $\gamma \in (0,1)$ is the discount rate. Q-value iteration is an operator $\mathcal{T}$ , whose domain is bounded real-valued functions over $S \times A$, defined as

$$ \mathcal{T} Q(s,a) = r(s,a) + \gamma \mathbb{E}_{P} \left[ \max_{a' \in A} Q(s',a') \right]$$

This operator is a contraction map in $$||\cdot ||_{\infty}$$, and so converges to a unique fixed point $$Q^*$$, where $$Q^*(s, a)$$ gives the expected value of the MDP starting from state $s$, taking action $a$, and thereafter following the optimal policy $$\pi^*(s) = \arg \max_{a \in A} Q^*(s,a)$$. 

A regret-minimisation problem considers a single decision-maker playing a game against an adversary. Consider a set of $n$ actions $a_1, \dots, a_n$ and the following game setup: 
* At time $t=0,1,\dots,T$: 
	* A decision maker picks a mixed strategy $\pi_t$ (probability distribution over its actions $A$).
	* An adversary picks a (bounded) cost vector, $c_t(a)$, for each action.
	* An action $a_t$ is chosen according to the distribution $\pi_t$ and the decision maker incurs a cost $$c_t(a_t)$$. The decision maker learns the entire cost vector $c_t$ not just the realised cost.  
	
We are not able to compare the cost of online decision-making to the cost of the best action sequence in hindsight - the benchmark of $$\sum^T_{t=0} \min_{a \in A}c_t(a)$$ is not achievable. Rather than comparing the expected cost of an algorithm to that of the best action *sequence* in hindsight, we compare it to the cost incurred by the best *fixed action* in hindsight. So, the benchmark becomes $$\min_{a\in A} \sum^T_{t=0} c_t(a)$$.  The time-averaged (external) regret of the action sequence $a_1, \dots, a_T$ with respect to the action $a$ is 

$$ \frac{1}{T+1} \left[ \sum_{t=0}^T c_t(a_t) - \sum_{i=0}^T c_t(a)  \right] $$


## No-Regret Learning

An algorithm is **no-regret** if there is a sequence of constants $\rho_t$ such that regardless of the adversary, the regret at time $t$ is at most $$\rho_t$$ and $$\lim_{t \rightarrow \infty} \rho_t = 0$$. A common bound is $O(1/\sqrt{k})$. An algorithm is **no-absolute-regret** if the regret is bounded by $||\rho_t||$, whereas it is **non-negative-regret** if regret is bounded by $(0,\rho_t)$.  


### Local No-Regret Learning (LONR) 

This is the algorithm proposed by Kash et al (2019). It is summarised as follows: 
* Initialise matrix $Q_0$ 
* Initialise $|S|$ copies of the no-absolute-regret algorithm (one for each state) and find the optimal policy $\pi_0$ for current state $s$. 
* Iteratively reveal costs to the copy of the algorithm for $s$ as $$c_t(a) = Q_t(s,a)$$, and update the policy $\pi_{t+1}$ according to the algorithm and the update rule

$$Q_{t+1} (s,a) = r(s,a) + \gamma \mathbb{E}_{P, \pi_k} \left[ Q_t(s',a') \right]$$


# Comments 

In the way it is currently presented, the above method is not suitable for a reinforcement learning environment since it assumes knowledge of transition probabilities, which an agent does not normally have (at least in model-free settings). In order to implement this in an MDP or Markov game setting with reinforcement learning, we have to make some adjustments to the specification. In order to implement, we need to specify a no-regret algorithm such as multiplicative weights or regret matching. 

The combination of Q-learning and no-regret has been proven only for a single agent in an MDP setting; the method has been shown experimentally to work in Markov games, but convergence is yet to be verified. 


# References
* Roughgarden T., CS364A: Algorithmic Game Theory Lecture #17: No-Regret Dynamics, [url](http://timroughgarden.org/f13/l/l17.pdf)
* Kash, I.A., Sullins, M. and Hofmann, K., 2019. Combining No-regret and Q-learning. *arXiv preprint arXiv:1910.03094.*
* Zinkevich, M., Johanson, M., Bowling, M. and Piccione, C., 2008. Regret minimization in games with incomplete information. In *Advances in Neural Information Processing Systems* (pp. 1729-1736).



