---
title: Implementation of Neural Processes
author: Marta Grzeskiewicz
date: 2020-03-06
tags: 
- Neural Processes
- Variational Inference
- Implementation
markup: mmark
---

# Introduction 

[Neural Processes](https://arxiv.org/abs/1807.01622) (NPs) aim to combine the computational efficiency and evaluation ability of neural networks (NNs) and the distributions over functions given by flexible and data-efficient but computationally-inefficient GPs. They do this by using an encoder-decoder structured network, where the encoder includes a latent distributional representation of the inputs. This gives the network a value over uncertainty and a distribution over possible functions. Using an aggregator in between the encoder and decoder, the computational efficiency is $O(n)$, where $n$ is the total number of data points. 


# Model 

The standard approach to defining a stochastic process is via its finite-dimensional marginal distributions. Specifically, we consider the process as a random function $F : X \rightarrow Y$ and for each finite sequence $x_{1:n} = (x_1, \dots , x_n)$ with $$x_i \in X$$ , we define the marginal joint distribution over the function values $$Y_{1:n} := (F(x_1), \dots , F(x_n))$$. For example, in the case of GPs, these joint distributions are multivariate Gaussians parameterised by a mean and a covariance function.

The original paper uses exchangeability and consistency to show that given a collection of joint distributions $\rho(x_{1:n})$ a stochastic process $F$ can be defined such that $\rho(x_{1:n})$ is the marginal distribution of $$(F(x_1), \dots , F(x_n))$$, for each finite sequence $x_{1:n}$. This result is used in the following set of equations. Given a particular instantiation of the stochastic process $f$, the joint distribution is defined as: 

$$\rho_{x_{1:n}} (y_{1:n}) = \int p(f) p(y_{1:n} | f, x_{1:n}) \text{d}f $$

Here $p$ denotes the abstract probability distribution over all random quantities. Instead of $Y_i = F(x_i)$, we add some
observation noise $$Y_i \sim \mathcal{N} (F(x_i), \sigma^2)$$ and define $p$ as:

$$p(y_{1:n} | f, x_{1:n}) = \Pi^n_{i=1} \mathcal{N}(y_i | f(x_i), \sigma^2)$$

Combining the above two equations, the stochastic process is specified by:

$$\rho_{x_{1:n}} (y_{1:n}) = \int p(f) \Pi^n_{i=1} \mathcal{N}(y_i | f(x_i), \sigma^2) \text{d}f$$

In other words, exchangeability and consistency of the collection of joint distributions $\{\rho(x_{1:n})\}$ imply the existence of a stochastic process $F$ such that the observations $Y_{1:n}$ become iid conditional upon $F$. In order to represent $F$ as a NP, it is first approximated by a NN and secondly it is assumed that $F$ can be parameterised by a high-dimensional random vector $z$, such that $F(x) = g(x, z)$, where $x$ is some input, for some fixed and learnable function $g$. This implies that the randomness is only because of $z$. The generative model then becomes 

$$ p(z, y_{1:n} | x_{1:n}) = p(z) \cdot \Pi^n_{i=1} \mathcal{N} \left( y_i | g(x_i, z), \sigma^2 \right)$$

where, in the nature of variational auto-encoders, $p(z)$ is multivariate standard normal and $g(x_i, z)$ is a NN.  

## Variational Inference

Here is where the context and target datasets come in: they are needed to differentiate between the variability of the random function from the variability of the datasets.  The dataset is split into a context set, $x_{1:m}, y_{1:m}$ and a target set $x_{m+1:n}, y_{m+1:n}$, and model the conditional of the target given the context. The authors use amortised variational inference to estimate the non-linear decoder $g$ and derive an [evidence lower bound (ELBO)](https://en.wikipedia.org/wiki/Evidence_lower_bound) which becomes the loss function for the network in implementation. Let $q(z|x_{1:n}, y_{1:n})$ be a variational posterior of the latent variables $z$, parameterised by another neural network that is invariant to permutations of the sequences $x_{1:n}, y_{1:n}$. It is given as 

$$\log p(y_{m+1:n} | x_{1:n}, y_{1:m})  \geq \mathbb{E}_{q(z | x_{1:n}, y_{1:n})} \left[ \sum_{i=m+1}^n \log p(y_i |z, x_i) + \log \frac{q(z | x_{1:m}, y_{1:m})}{q(z|x_{1:n}, y_{1:n})} \right] $$

where the variational posterior $q(z | x_{1:m}, y_{1:m})$ approximates the conditional prior $p(z|x_{1:m}, y_{1:m})$.

## Global Latent Variable 

We have assumed above that there is a latent variable $z$ that captures the stochastic process $F$. This variable captures global uncertainty, which allows us to sample at the global level - one function $f_d$ at a time, rather than at a local output level - one $y_i$ value for each $x_i$ at a time (independently of the remaining $y_T$). This corresponds to **two encoders** of the same structure - a stochastic one based on the distribution of latent variable $z$ and a deterministic one based on the input $x_i$. Passing all of the context’s information through this single variable, means we can formulate the model in a Bayesian framework. We first obtain the prior $p(z)$ learned during training. Adding observations means we can perform a Bayesian update and obtain the posterior $p(z|C)$ over the function given the context. 

## Model summary

NPs are made of three general parts: 
* encoder $h$ from input space into representation space that takes in pairs of $(x,y)_i$ context values and produces a representation $r_i = h((x,y)_i)$ for each pair. It is parameterised by a neural network.
* aggregator $a$ that summarises the encoded inputs. The goal is to obtain a single order-invariant global representation $r$ that parameterises the latent distribution $$ Z \sim \mathcal{N}(\mu(r), \textbf{I}\sigma(r))$$ , for example, the mean function, $$r = a(r_i) = \frac{1}{n} \sum_{i=1}^n r_i$$.
* conditional decoder $g$ that takes as input the sampled global latent variable $z$ as well as the new target locations $x_T$ and outputs the predictions $\hat{y}_T$ for $f(x_T) = y_T$.


# Implementation 

We implement the NP with Tensorflow (TF) and TensorflowProbability (TFP). First, we define the structure of the neural network which parameterises the encoder. In this implementation, we have assumed the same structure for the encoder and decoder. This can easily be adjusted to specify a different structure in a `get_decoder` function of the same structure, with a different number of layers, for example. We have chosen to implement two dense layers with 32 neurons each, with the first including a non-linear activation function [(ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). 

```python
def get_encoder():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation="relu"),
                                        tf.keras.layers.Dense(32, activation=None)])
    return model

```

This is used in the encoder and the decoder when initialising the model. The encoder is now split into two: the deterministic part and the latent part, due to the presence of the global latent variable.  

```python
class NPModel(tf.keras.Model):
    def __init__(self, x_context, y_context, x_target):
        super(NPModel, self).__init__()
        self._latent_encoder = get_encoder()
        self._deterministic_encoder = get_encoder()
        self._decoder = get_encoder()
        self.x_context = x_context
        self.y_context = y_context
        
```

We also define the aggregator, used to obtain the global latent variable that captures $F$. We assume that the aggregator is the mean, but this can be replaced by some other function that gives single order-invariant value. 

```python 
    def conglomerate(self, tensor):
        return tf.reduce_mean(tensor, axis=0, keepdims=True)
        
```

Next, we define the call function, which takes the target input values and runs them through the NP to obtain a prediction for the target output. We first use the trained parameters of encoders to obtain the concatenated latent and deterministic representations of the input, which are then decoded using the trained decoder. 

```python

    def call(self, x_target):
        params_context = self.conglomerate(self._latent_encoder(self.x_context, self.y_context))
        mu, log_sigma = tf.split(params_context, 2, axis=1)
        sigma = tf.exp(log_sigma)
        
        latent_rep = tf.random.normal(mu.shape) * sigma + mu 
        deterministic_rep = self.conglomerate(self._deterministic_encoder(self.x_context, self.y_context))
        
        representation = tf.concat([deterministic_rep, latent_rep],axis=1)
        tiled = representation * tf.ones_like(x_target)
        
        params_decoder = self._decoder(tf.concat([tiled, x_target], axis=1))
        return tf.concat([params_decoder, x_target], axis=1)
        
```

The line `latent_rep` uses the reparametarisation trick, which allows for gradient flow in TF. This makes the latent representation, a probability distribution, differentiable in both $\mu$ and $\log \sigma$. If we were to instead specify `tf.random.normal` with the values as inputs, TF would not be able to differentiate and the network would not be able to train.  

The loss function is given by the ELBO that was derived earlier. It is given by

$$ \mathcal{L} = - \log p(y_t|\hat{y}_T) + \text{KL}(z_T || z_C)$$

where in general terms, 
$$\text{KL}(P\parallel Q)=\int _{-\infty }^{\infty } P(x)\log \left({\frac {P(x)}{Q(x)}}\right) \text{d}x$$. See the [post on variational inference](https://martagrz.github.io/post/variational-inference/) to see how this was derived.

The first term in the loss function is the likelihood of the true target output given the predicted target output. We want to maximise this likelihood, meaning that our network obtains close to true values. This is regularised by the KL divergence, since we want the distribution of the target and context variables to be the same since they come from the same dataset. 

```python
    def loss(self,y_target,output):
        params_decoder = output[:, :32]
        x_target = output[:, 32:]
        mu, log_sigma = tf.split(params_decoder, 2, axis=1)
        sigma = tf.exp(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(mu, sigma)
        log_p = dist.log_prob(y_target)        
        
        #Context
        params_context = self.conglomerate(self._latent_encoder(self.x_context, self.y_context)) 
        mu_context, log_sigma_context = tf.split(params_context, 2, axis=-1)
        sigma_context = tf.exp(log_sigma_context)
        prior = tfp.distributions.Normal(loc=mu_context, scale=sigma_context)
        
        #Target
        params_target = self.conglomerate(self._latent_encoder(x_target, y_target)) 
        mu_target, log_sigma_target = tf.split(params_target, 2, axis=-1)
        sigma_target = tf.exp(log_sigma_target)
        posterior = tfp.distributions.Normal(loc=mu_target, scale=sigma_target)
        
        kl = tf.reduce_sum(tfp.distributions.kl_divergence(posterior, prior),axis=-1, keepdims=True)
        loss = - tf.reduce_mean(log_p - kl /tf.cast(100,tf.float32))
        return loss
        
```

The performance is measured by the root mean squared error between the target values predicted by the NP and the true target values. 

```python
    def rms(self, y_target, output):
        params_decoder = output[:,:32]
        mu, log_sigma = tf.split(params_decoder,2,axis=1)
        mse = tf.reduce_mean(tf.math.square(y_target - mu))
        return tf.math.sqrt(mse)

```

That is it! Those are all the parts required to implement the NP. See [github](https://github.com/martagrz/neural-processes/tree/master/neural-processes) for the full implementation. 


# References 
* Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D.J., Eslami, S.M. and Teh, Y.W., 2018. Neural processes. arXiv preprint arXiv:1807.01622.

