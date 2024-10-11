---
title: "Reinforcement Learning"
draft: false
comments: true
date: "2024-10-02"
slug: "reinforcement-learning"
tags: ["rl", "reinforcement-learning", "deep reinforcement learning", "policy gradient"]
categories: [reinforcement learning, policy gradient]
showpagemeta: yes
showcomments: true
author: "Cristóbal Alcázar"
description: "My notes on RL and policy gradient methods"
keywords:
    - "Reinforcement Learning"
    - "Policy Gradient"
bibliography: "rl.bib"
---

<details>
    <summary><b>Table of Contents</b></summary>
    <ul>
        <li><a href="#the-framework-for-learning-to-act">The Framework for Learning to Act</a></li>
        <li><a href="#policy-optimization">Policy Optimization</a>
            <ul>
                <li><a href="#learning-the-policy">Learning the Policy</a></li>
                <li><a href="#gradient-estimation-via-score-function">Gradient Estimation via Score Function</a></li>
            </ul>
        </li>
        <li><a href="#vanilla-policy-gradient-aka-reinforce">Vanilla Policy Gradient, aka REINFORCE</a></li>
        <li><a href="#actor-critic-methods">Actor-Critic Methods</a></li>
        <li><a href="#references">References</a></li>
    </ul>
</details>

<br>

Reinforcement learning (RL) <a href="http://incompleteideas.net/book/the-book-2nd.html" target="_blank">(Sutton, 1998)</a> is all about the interaction between an agent and its environment, where learning occurs through trial-and-error. The agent observes the current state of the environment, takes actions based on these observations, and influences new possible state configurations while receiving rewards based on its actions. The primary objective is to maximize cumulative rewards, which drives the agent's sequence of decisions towards achieving specific goals, such as escaping from a maze, <a href="https://arxiv.org/abs/1312.5602" target="_blank">winning an Atari (Mnih. 2013) </a>, or <a href="https://deepmind.google/technologies/alphago/" target="_blank">defeating the world champion of Go (Silver, 2016)</a>. But how does the agent learn to act effectively to achieve its goal? RL algorithms are designed to maximize the total rewards obtained by the agent, thereby guiding its actions towards these objectives. 

In this post, we will introduce the essential concepts of RL required to implement these agents. We will specifically focus on model-free RL, where the agent learns to act without constructing a model of its environment, as opposed to model-based RL, which involves such modeling. The goal is to design agents that learn to perform well solely by consuming experiences from their environment. By understanding the fundamentals of designing such agents, we will explore policy optimization methods, such as REINFORCE and PPO, which are used to refine the agent’s behavior.  

With the knowledge gained from this chapter, we will be equipped to set-up and implement this framework under popular research environment such as ATARI pong.

<!-- <b>Table of Contents:</b>
- [The Framework for Learning to Act](#the-framework-for-learning-to-act)
- [Policy Optimization](#policy-optimization)
  - [Learning the Policy](#learning-the-policy)
  - [Gradient Estimation via Score Function](#gradient-estimation-via-score-function)
- [Vanilla Policy Gradient, aka REINFORCE](#vanilla-policy-gradient-aka-reinforce)
- [Actor-Critic Methods](#actor-critic-methods)
- [References](#references) -->


## The Framework for Learning to Act

The starting point for designing agents that learn to act is the Markov Decision Process (MDP) framework \cite{Sutton1998}. An MDP is a mathematical object that describes the interaction between the agent and the environment. This interaction is characterized by a tuple \\(\langle \mathcal{S}, \mathcal{A}, P, R, \rho_{0}, \gamma \rangle\\), where:


* \\(\mathcal{S}\\), **state space**, set of possible states in the environment.
* \\(\mathcal{A}\\), **action space**, set of possible actions available to the agent.
* \\(P: \mathcal{S}\times\mathcal{A}\rightarrow\Delta(\mathcal{S})\\), **transition probability distribution**, which gives the probability of the environment for transitioning to a new state \\(s_{t+1}\\) with a reward \\(r_t\\) given the current state \\(s_{t}\\) and action \\(a_{t}\\).
* \\(R: \mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}\\), **reward function**, which provides a scalar feedback signal \\(r_{t}\\) (aka reward) to the agent after taking an action \\(a_{t}\\) and reaching the subsequent state \\(s_{t+1}\\).
* \\(\rho_{0}\\), **initial state distribution**, which determines the probability of the agent starting in a particular state.
* \\(\gamma\in\left[0, 1 \right]\\) is the **discount factor**, which determines the importance of future rewards.

<!-- \begin{figure}[ht]
    \centering
    \includegraphics[scale=0.63]{ch3-rl/MDP-diagram.pdf}
    \captionsetup{width=\textwidth} % set the width of the caption
    \caption{\textbf{Left:} A loop representation of a Markov Decision Process (MDP). \textbf{Right:} An unrolled MDP depecting an episodic case with a finite horizon $T$ and a parameterized policy $\pi_{\theta}$.}
    \label{fig:mdp-diagram}
  \end{figure} -->


<br>

<figure id="fig:mdp-diagram">
  <img src="https://alkzar.cl/img/rl/MDP-diagram.png" alt="Markov Decision Process Diagram">
  <figcaption><br><small>Figure 1. <b>Left:</b> A loop representation of a Markov Decision Process (MDP). <b>Right:</b> An unrilled MDP depecting an episodic case with a finite horizon \\(T\\) and a parameterized policy \\(\pi\_{\theta}\\).</small></figcaption>
</figure>

 Markov Decision Processes generate sequences of state-action pairs, or trajectories \\(\tau\\), starting from an initial state \\(s_{0}\sim\rho_{0}\\). The agent's behavior is determined by a policy \\(\pi:\mathcal{S}\rightarrow\Delta(\mathcal{A})\\), which maps states to a probability distribution over actions. An action \\(a_{0}\sim\pi(s_{0})\\) is chosen, leading to the next state \\(s_{1}\\) according to the transition distribution \\(P\\), and a reward \\(r_{0}=R(a_{0}, s_{0})\\) is received. This cycle repeats iteratively, with the agent selecting actions, transitioning through states, and receiving rewards, as shown on the left side of [Figure 1](#fig:mdp-diagram). Thus, the trajectory \\(\tau\\) encapsulates the dynamic sequence of state-action pairs resulting from the agent's interaction with its environment. 

 The process can continue indefinitely, known as an infinite horizon, or be confined to episodes that end in the terminal state \\(s_{T}\\), referred to as episodic tasks, such as winning or losing a game, as illustrated on the right side of [Figure 1](#fig:mdp-diagram). It is important to note that the transition to the next state depends only on the current state and action, not on the sequence of prior events. This characteristic is known as the _Markov property_, which states that the future and the past are conditionally independent, given the present (_memoryless_). In this work, we focus on the episodic setting, where the trajectory begins at \\(s_{0}\\) and concludes at \\(s_{T}\\), with a finite horizon \\(T\\). Therefore, the trajectory \\(\tau\\) is defined as \\(\tau = (s_{0}, a_{0}, \dots, s_{T-1}, a_{T-1}, s_{T})\\), summarizing the agent's behavior throughout the episodic task. 

 In reinforcement learning, the primary goal is for the agent to develop a behavior that maximizes the expected return from its actions results within the environment. This concept of maximization is formalized through the objective function \\(\mathcal{J}\_{\text{RL}}(\theta)\\), which aims to maximize the expected return over a collection of trajectories \\( \{\tau^{(i)}\}\_{1:N} \\) generated by the policy \\(\pi\\), commonly referred to as "policy rollouts". The term "rollout" is used to describe the process of simulating the agent's behavior in the environment by executing the policy \\(\pi\\) and observing the resulting trajectory \\(\tau\\). The objective function is defined as follows:

\\[
    \begin{equation}
        \mathcal{J}\_{\text{RL}}=\underset{\pi}{\text{maximize }} \mathbb{E}\_{\tau\sim\pi}\left[R(\tau)\right]
    \end{equation}
\\]

The return over a trajectory \\(\tau\\) is defined as the accumulated discounted rewards of the trajectory, \\(R(\tau) = \sum_{t=0}^{T-1}\gamma^{t}r_{t}\\). The reward signals \\(r_{t}\\) are the inmmediate effect of taking the actions, and the return is the cumulative rewards obtained during the trajectory, considering a discount factor \\(\gamma\\), which gives more importance to the rewards of nearer actions than to future rewards. 


## Policy Optimization

In reinforcement learning there are different approaches to solve the MDP formulated in the previous section, which are summarized in [Figure 2](#fig:rl-model-free-taxonomy). The most common are value-based methods and policy-based methods. In value-based methods, the agent learns which state is more valuable and take action that leads to it. In policy-based methods, the agent learns a policy that directly maps states to actions. In this work we will focus on the latter methods, specifically in policy gradients. 

<figure id="fig:rl-model-free-taxonomy">
  <img src="https://alkzar.cl/img/rl/rf-solve-methods-schulman-thesis-img.png" alt="Markov Decision Process Diagram">
  <figcaption>Figure 2. <b>Illustration of a taxonomy of model-free RL algorithms.</b> Source: <a href="https://rail.eecs.berkeley.edu/deeprlcourse/" target="_blank">Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs by John, Schulman (2016)</a> \cite{schulman2016optimizing}.</figcaption>
</figure>

Other approaches for finding a policy is by non solving the MDP, but by directly optimizing the policy. This is the case of derivative free optimization (DFO), or evolutionary algorithms, in which the policy is parameterized by a vector \\(\theta\\), and the agent explores the space of parameters by searching. Nothing of the temporal structure and actions of the MDPs are considered in this kind of solution.

Policy gradient methods provide a way to reduce reinforcement learning to stochastic gradient descent, by providing a connection between how function approximation is solved in supervised learning settings, but with the key diffrence that the dataset is collected using the model itself plus a reward signal that acts as a "label". 

### Learning the Policy

The starting point is to think of trajectories as units of learning instead of individual observations (i.e., actions). What dynamics generate a trajectory? 
Given a policy \\(\pi_{\theta}\\), represented as a function with parameter \\(\theta\in \mathbb{R}^{d}\\), whose input is a representation of the state and whose output is action selection probabilities, we can deploy the agent into its environment at an initial state \\(s_0\\) and observe its actions in inference mode or _evaluation phase_ \citep{sutton1999policy}. The agent continuously promotes actions based on the current state \\(s_{t}\\) until the episode ends in a terminal state, when \\(t=T\\). At this point, we can determine if the goal was accomplished, such as winning the ATARI Pong game, <a href="https://github.com/alcazar90/ddpo-celebahq" target="_blank"><i>or generating aesthetically pleasing samples from a diffusion model</i></a>. 
The returns are the scalar value that assets perfomance whether we have achieved the ultimate goal, effectively acting as a "proxy" of a label for the overall trajectory. Thus, the trajectory serves as our unit of learning, and the remaining task is to establish the feedback mechanism for the _learning phase_. 

 Intuitivelly, we want to collect the trajectories and make the good trajectories and actions more probable, and push the actions towards betters actions. 

 Mathematically, we aim to perform stochastic optimization to learn the agent’s parameters. This involves obtaining gradient information from sample trajectories, with performance assessed by a scalar-value function (i.e. reward). The optimization is stochastic because both the agent and the environment contain elements of randomness, meaning we can only compute estimates of the gradient. Crucially, we are estimating the gradient of the expected return with respect to the policy parameters. To address this, we employ Monte Carlo Gradient Estimation \citep{mohamed2020monte}, specifically using the score function method. From a machine learning perspective, this involves dealing with the stochasticity of the gradient estimates, \\(\hat{g}\\), and using gradient ascent algorithms to update the policy parameters based on these estimates, along with a learning rate \\(\alpha\\) to control the step size of the optimization process,

\begin{equation}\label{eqn:gradient-ascent}
    \theta \leftarrow \theta + \alpha \hat{g}_{N}.
\end{equation}


### Gradient Estimation via Score Function

The gradient estimation can be obtained using the score function gradient estimator. Let's introduce the following probability objective \\(\mathcal{F}\\), defined in the <a href="https://en.wikipedia.org/wiki/Ambient_space_(mathematics)" target="_blank">ambient space</a> \\(\mathcal{X}\in\mathbb{R}^n\\) and with parameters \\(\theta\in\mathbb{R}^n\\),

\\[
    \begin{equation}\label{eqn:probability-objective}
    \mathcal{F}(\theta) = \int\_{\mathcal{X}} p(\mathrm{x; \theta})f(\mathrm{x})~d\mathrm{x} = \mathbb{E}\_{p(\mathrm{x};\theta)}\left[f(\mathrm{x})\right].
    \end{equation}
\\]

Here, \\(f\\) is a scalar-valued function, similar to how the reward is represented in the reinforcement learning setting. The _score function_ is the derivative of the log probability distribution  \\(\nabla_{\theta}\log p(\mathrm{x};\theta)\\) with respect to its parameters \\(\theta\\). We can use the following identity to establish a connection between the score function and the probability distribution \\(p(\mathrm{x};\theta)\\),

\begin{equation}\label{eqn:log-derivative-trick-expression}
    \begin{split}
        \nabla_\theta\log p(\mathrm{x};\theta) &= \frac{\nabla_{\theta}p(\mathrm{x}; \theta)}{p(\mathrm{x};\theta)} \\\\
        p(\mathrm{x};\theta) \nabla_{\theta}\log p(\mathrm{x};\theta) &= \nabla_{\theta}p(\mathrm{x};\theta).
    \end{split}
\end{equation}

Therefore, taking the gradient of the objective \\(\mathcal{F}(\theta)\\) with respect to the parameter \\(\theta\\), we have

\\[
    \begin{equation}\label{eqn:score-function-gradient-objective}
        \begin{split}
            g = \nabla\_{\theta} \mathbb{E}\_{p(\mathrm{x};\theta)}\left[f(\mathrm{x})\right] &= \nabla\_{\theta}\int\_{\mathcal{X}} p(\mathrm{x};\theta) f(\mathrm{x}) d\mathrm{x} \\\\
            &= \int\_\mathcal{X} \nabla\_{\theta}~p(\mathrm{x};\theta)f(\mathrm{x})d\mathrm{x} \\\\
            &= \int\_{\mathcal{X}} p(\mathrm{x};\theta)\nabla\_{\theta}\log p (\mathrm{x}; \theta) f(\mathrm{x})d\mathrm{x} \\\\
            &= \mathbb{E}\_{p(\mathrm{x};\theta)}\left[f(\mathrm{x})\nabla\_{\theta} \log p(\mathrm{x};\theta) \right]
        \end{split}
    \end{equation}
\\]


The use of the log-derivative rule on the above equation to introduce the score function is also known as the <a href="https://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/" target="_blank">_log-derivative trick_</a>. Now, we can compute an estimate of the gradient, \\(\hat{g}\\), using Monte Carlo estimation with samples from the distribution \\(p(\mathrm{x};\theta)\\) as follows:

\\[
    \begin{equation}\label{eqn:score-function-gradient-estimator}
        \hat{g}\_{N} = \frac{1}{N}\sum\_{i=1}^{N}f\left(\hat{\mathrm{x}}^{(i)}\right) \nabla\_{\theta}\log p\left(\hat{\mathrm{x}}^{(i)};\theta\right).
    \end{equation}
\\]


We draw \\(N\\) samples \\(\hat{\mathrm{x}}\sim p(\mathrm{x};\theta)\\), compute the gradient of the log-probability for each sample, and multiply by the scalar-valued function \\(f\\) evaluated at the sample. The average of these terms is an unbiased estimate of the gradient of the objective \\(g\\), which we can use for gradient ascent.

There are two important points to mention about the previous equation.

* The function \\(f\\) can be any arbitrary function we can evaluate on \\(\mathrm{x}\\). Even if \\(f\\) is non-differentiable with respect to \\(\theta\\), it can still be used to compute the gradient estimation \\(\hat{g}\\).
* The expectation of the score function is zero, meaning that the gradient estimator is unbiased

\\[
    \begin{equation}\label{eqn:score-function-expectation-zero}
    \begin{split}
        \mathbb{E}\_{p(\mathrm{x};\theta)}\left[\nabla\_{\theta}\log p(\mathrm{x};\theta)\right] 
        &= \int\_{\mathcal{X}}p(\mathrm{x};\theta)\nabla_{\theta}\log p(\mathrm{x}; \theta) d\mathrm{x} \\\\
        &= \int\_{\mathcal{X}} p(\mathrm{x};\theta)\frac{\nabla\_{\theta} p(\mathrm{x}; \theta)}{p(\mathrm{x};\theta)}d\mathrm{x} \\\\
        &= \int\_{\mathcal{x}}\nabla\_{\theta}p(\mathrm{x};\theta)d\mathrm{x} \\\\
        &= \nabla\_{\theta}\int\_{\mathcal{X}} p(\mathrm{x}; \theta)d\mathrm{x} = \nabla\_{\theta} 1 =0.
    \end{split}
    \end{equation}
\\]


The last point is particularly useful because we can replace \\(f\\) with a shifted version given a constant \\(\beta\\), and still obtain an unbiased estimate of the gradient, which can be beneficial for the optimization task:

\\[
\hat{g}\_{N} = \mathbb{E}\_{p(\mathrm{x}\_{\theta})}\left[(f(\mathrm{x}) - \beta) \nabla\_{\theta} \log p(\mathrm{x}; \theta)\right].
\\]

Using a ***baseline function*** to determine \\(\beta\\), that does not depend on the parameter 
\\(\theta\\), can reduce the variance of the estimator \citep{mohamed2020monte}. The baseline function, 
which satisfies the property that the score function expectation is zero, can be any function independent of \\(\theta\\). When a baseline is chosen to be close to the scalar-valued function \\(f\\), it effectively reduces the variance of the estimator. This reduction in variance helps stabilize the updates by minimizing fluctuations in the gradients estimates, leading to more reliable and efficient learning.

## Vanilla Policy Gradient, aka REINFORCE

The REINFORCE algorithm \citep{williams1992simple} translates the previous 
derivation of gradient estimation via the score function into reinforcement learning terminology. This is the earliest member of the Policy Gradient family (Figure~\ref{fig:rl-model-free-taxonomy}), where the objective is to maximize the expected return of the trajectory \\(\tau\\) under a policy \\(\pi\\) parameterized by \\(\theta\\) (e.g., a neural network). At each state \\(s_{t}\\), the agent takes an action \\(a_{t}\\) according to the policy \\(\pi\\), which generates a probability distribution over actions \\(\pi(a_{t}\mid s_{t};\theta)\\). Here, we will use the notation \\(\pi_{\theta}(\cdot)\\) instead of \\(\pi(\cdot;\theta)\\). 

 As we mentioned in previous section, a trajectory \\(\tau\\) represents the sequence of state-action pairs resulting from the agent's interaction with its environment. From the initial state \\(s_{0}\\) to the terminal state \\(s_{T}\\), the trajectory \\(\tau\\) is a sequence of states and actions, \\(\tau = (s_{0}, a_{0}, \dots, s_{T-1}, a_{T-1}, s_{T})\\), which describes how the agent acts during the episodic task. Let \\(p_{\theta}(\tau)\\) be the
probability of obtaining the trajectory under the policy \\(\pi_{\theta}\\). 

 We thus have a distribution of trajectories. Remember that the trajectory \\(\tau\\) is the learning unit for our policy \\(\pi_{\theta}\\), as it
tells us if the consequences of each action led to a favorable final outcome on the terminal state \\(s_{T}\\) (e.g. win/lose). The goal is to maximize the exptected return of the trajectories on average, and the return \\(R(\tau)\\) could be the cumulative rewards obtained during the ***episode*** or the discounted rewards. The expected return is given by the following expression:

\\[
    \begin{equation}\label{eqn:rl-objective}
        \mathcal{J}(\theta)\_{\text{RL}}=\mathbb{E}\_{\tau\sim p\_{\theta}(\tau)}\left[R(\tau)\right].
    \end{equation}
\\]

This is the objective we want to maximize, which is a particular case of Equation~(\ref{eqn:probability-objective}) with the scalar-valued function \\(f(\mathrm{x}) = R(\tau)\\), representing the return of the trajectory. Let's use the techniques from the previous section to compute the gradient of the objective in Equation~(\ref{eqn:rl-objective}) with respect to the policy parameter \\(\theta\\). The gradient estimation is given by:

\\[
    \begin{equation}\label{eqn:rl-gradient-estimator-vanilla}
        \nabla\_{\theta} \mathbb{E}\_{\tau\sim p\_{\theta}(\tau)}[R(\tau)] = \mathbb{E}\_{\tau\sim p\_{\theta}(\tau)}\left[R(\tau)\nabla\_{\theta}\log p\_{\theta}(\tau)\right].
    \end{equation}    
\\]

What is \\(p_{\theta}(\tau)\\) exactly? Given that the trajectory is a sequence of states and actions, and assuming the Markov property imposed by the MDP, the probability of the trajectory is defined as follows:

\\[
    \begin{equation}\label{eqn:trajectory-probability-expandedi}
        \begin{split}
            p\_{\theta}(\tau) &= p\_\theta(s\_{0}, a\_{0}, s\_{1}, a\_{1}, \dots, s\_{T-1}, a\_{T-1}, s\_{T}) \\\\
            &= \rho(s\_0)~\prod\_{t=0}^{T-1} \pi\_{\theta}(a\_{t}\mid s\_{t})~P(s\_{t+1}, r\_{t}\mid a\_{t}, s\_{t}).
        \end{split}
    \end{equation}
\\]

In the above expression, \\(\rho(s_{0})\\) denotes the distribution of initial states, while \\(P(s\_{t+1}, r\_{t}\mid a\_{t}, s\_{t})\\) represents the transition model, which updates the environment context based on the action \\(a\_{t}\\) taken in the current state \\(s\_{t}\\). A crucial step in estimating the gradient is computing the logarithm of the trajectory probability. Following this, we calculate the gradient with respect to the policy parameter \\(\theta\\),

\begin{equation}\label{eqn:trajectory-gradient-score}
    \begin{split}
        \log p\_{\theta}(\tau) &= \log \rho(s_0) + \sum\_{t=0}^{T-1}\log \pi_{\theta}(a_{t}\mid s\_{t}) + \log P(s\_{t+1}, r\_{t}\mid a\_{t}, s\_{t}) \\\\
        \nabla\_{\theta}\log p\_{\theta}(\tau) &= \log \nabla\_{\theta}\rho(s\_0) + \sum\_{t=0}^{T-1}\nabla\_{\theta}\log \pi\_{\theta}(a\_{t}\mid s\_{t}) + \log\nabla\_{\theta} P(s\_{t+1}, r\_{t}\mid a\_{t}, s\_{t}) \\\\
        \nabla\_{\theta} \log p\_{\theta}(\tau) &=  \sum\_{t=0}^{T-1}\nabla\_{\theta}\log \pi\_{\theta}(a\_{t}\mid s\_{t}).
    \end{split}
\end{equation}

The distribution of initial states and the transition probabilities are disregarded because they are independent of \\(\theta\\), thereby simplifying significantly the computations needed for gradient estimation. By substituting the final expression from Equation~(\ref{eqn:trajectory-gradient-score}) into the gradient estimation of the objective in Equation~(\ref {eqn:rl-gradient-estimator-vanilla}), we derive the REINFORCE gradient estimator

\\[
    \begin{equation}\label{eqn:reinforce-gradient-estimator}
        \begin{split}
            g &= \nabla\_{\theta}\mathbb{E}\_{\tau\sim p\_{\theta}(\tau)}[R(\tau)] \\\\
            &= \mathbb{E}\_{\tau\sim p\_{\theta}(\tau)}\left[\sum\_{t=0}^{t-1} \nabla\_{\theta}\log \pi\_{\theta} (a\_t\mid s\_t) R(\tau)\right]  \\\\
            \hat{g} &= \frac{1}{\mid\mathcal{D}^{\pi\_{\theta}}\mid}\sum\_{\tau\in\mathcal{D}^{\pi\_{\theta}}}\left[~\sum\_{t=0}^{t-1} \nabla\_{\theta} \log\pi\_{\theta} (a\_{t}\mid s\_{t}) R(\tau) \right].
        \end{split}
    \end{equation}
\\]

The core concept is to collect a set of trajectories \\(\mathcal{D}^{\pi_{\theta}}\\) under the policy \\(\pi_{\theta}\\) and update the policy parameters \\(\theta\\) to increase the likelihood of high-reward trajectories while decreasing the likelihood of low-reward ones, as illustrated in Figure~\ref{fig:anatomy-rl-trajectories}. This trial-and-error learning approach, described in [Algorithm 1](#alg:reinforce), repeats this process over multiple iterations, reinforcing successful trajectories and discouraging unsuccessful ones, thus encoding the agent's behavior in its parameters. 

<!-- % algoritmo naive REINFORCE -->
<div id="alg:reinforce">
    <big><b>Algorithm 1: Vanilla Policy Gradient, aka REINFORCE</b></big>
    <ol>
        <li>Initialize policy \( \pi_{\theta} \), set learning rate \( \alpha \)</li>
        <!-- The commented out line can be included or excluded as needed -->
        <!-- <li>Generate \( \tau=(s_0, a_0, ..., s_{T-1}, a_{T-1}, s_{T}) \) by sampling from current \( \pi_{\theta} \)</li> -->
        <li>For \( \text{iteration}=0, 1, 2, \dots, N \):
            <ol>
                <li>Collect a set of trajectories \( \mathcal{D}^{\pi_{\theta}}=\{\tau^{(i)}\} \) by sampling from the current policy \( \pi_{\theta} \)</li>
                <li>Calculate the returns \( R(\tau) \) for each trajectory \( \tau\in\mathcal{D}^{\pi_{\theta}} \)</li>
                <li>Update the policy: \( \theta \leftarrow \theta + \alpha \left(\frac{1}{|\mathcal{D}^{\pi_{\theta}}|}\sum_{\tau\in\mathcal{D}^{\pi_{\theta}}}\left[\sum_{t=0}^{T-1}\nabla_{\theta}\log\pi_{\theta}(a_{t}| s_{t})R(\tau)\right]\right) \)</li>
            </ol>
        </li>
    </ol>
</div>

 **Reducing the variance of the estimator**. Using two techniques,
 <a href="https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you" target="_blank">reward-to-go</a> and _baseline_, we can improve the quality of the gradient estimator in Equation~(\ref{eqn:reinforce-gradient-estimator}). 


<figure id="fig:anatomy-rl-trajectories" style="text-align: center;">
  <img src="https://alkzar.cl/img/rl/simulated-trajectories-levine-slides.png" alt="Simulated trajectories levine slides">
  <figcaption>
    <b>Illustration of three simulated trajectories</b>, denoted as \\(\\\{\tau^{(i)}\\\}\\) where \\(i=(1,2,3)\\), traversing the parametric space \\(\theta\in\mathbb{R}^2\\) under the policy \\(\pi\_{\theta}\\). Each trajectory is marked with a colored symbol (cross, check) representing its _goodness_ based on the reward function \\(R(\tau^{(i)})\\). <b>Source:</b> <a href="https://rail.eecs.berkeley.edu/deeprlcourse/" target="_blank">Policy Gradients Lecture, Deep Reinforcement Learning Course</a> by Sergey Levine.
  </figcaption>
</figure>
  
The reward-to-go technique is a simple trick that can reduce the variance of the gradient estimator by taking advantage of the _temporal structure_ of the problem. The idea is to weight the gradient of the log-probability of an action \\(a_{t}\\) by the sum of rewards from the current timestep \\(t\\) to the end of the trajectory \\(T-1\\). This way, the gradient of the log-probability of an action is only weighted by the consequence of that action on the future rewards, removing terms that do not depend on \\(a_{t}\\). Let's introduce this technique by using the gradient estimation in Equation~(\ref{eqn:reinforce-gradient-estimator}) and replacing \\(R(\tau)\\) naively using the sum of total trajectory reward **\footnote**{The same applies for discounted returns or other kind of returns \\(R(\tau)\\).}

\\[
    \begin{equation}\label{eqn:reinforce-gradient-reward-to-go}
        \begin{split}
            \hat{g} &= \frac{1}{\mid\mathcal{D}^{\pi\_{\theta}}\mid}\sum\_{\tau\in\mathcal{D}^{\pi\_{\theta}}}\left[\sum\_{t=0}^{T-1} \nabla\_{\theta} \log\pi\_{\theta} (a\_{t}\mid s\_{t})\sum\_{t=0}^{T-1} r\_{t}\right] \\\\
                    &= \frac{1}{\mid\mathcal{D}^{\pi\_{\theta}}\mid}\sum\_{\tau\in\mathcal{D}^{\pi\_{\theta}}}\left[~\sum\_{t=0}^{T-1} \nabla\_{\theta} \log\pi\_{\theta} (a\_{t}\mid s\_{t}) \left( \sum\_{t=0}^{t-1} r\_{t}  + \sum\_{t'=t}^{T-1} r\_{t'} \right)\right] \\\\
                    &= \frac{1}{\mid\mathcal{D}^{\pi\_{\theta}}\mid} \sum\_{\tau\in\mathcal{D}^{\pi\_{\theta}}} \left[ ~\sum\_{t=0}^{T-1} \nabla\_{\theta} \log\pi\_{\theta} (a\_{t}\mid s\_{t}) \sum\_{t'=t}^{T-1} r\_{t'}\  \right] .
        \end{split}
    \end{equation}
\\]


As we saw at the end of Section~\ref{sec:gradient-estimation-score-function}, it is possible to reduce the variance of the gradient estimator by using a baseline function, \\(b(s_{t})\\), without biasing the estimator. However, is the expectation of the score still unbiased in this setting? 

\\[
\begin{equation}\label{eqn:reinforce-gradient-estimator-baseline}
    \begin{split}
        \nabla\_{\theta}\mathbb{E}\_{\tau\sim p\_{\theta}(\tau)} &= \mathbb{E}\_{\tau\sim p\_{\theta}(\tau)} \left[\sum\_{t=0}^{T-1}\nabla\_{\theta}\log\pi\_{\theta}(a\_{t}\mid s\_{t})  \left(\sum\_{t'=t}^{T-1} r\_{t'}-b(s\_{t'}) \right)\right].
    \end{split}
\end{equation}
\\]


The proof follows a similar argument as shown in Equation~(\ref{eqn:score-function-expectation-zero}), with the key difference being that the expectation is taken with respect \\(p_{\theta}(\tau)\\), which is a sequence of random variables. By leveraging the linearity of the expectation property, we can focus on a single term at step \\(t\\) of Equation~(\ref{eqn:reinforce-gradient-estimator-baseline}) to demonstrate that the baseline does not affect the expectation of the score function. We split the trajectory sequence \\(\tau\\) at step \\(t\\) into: \\(\tau_{0:t}\\) and \\(\tau_{t+1:T-1}\\), and then expand it into state-action pairs **\footnote**{A criterion used when splitting the trajectory is that state-action pairs are formed given that \\(s_{t}\\) is a consequence of action \\(a_{t-1}\\), and taking action \\(a_{t}\\) results in state \\(s_{t+1}\\). Notice both expectations from step 1 and 2 in Equation~(\ref{eqn:reinforce-baseline-unbiased}).}

\\[
\begin{equation}\label{eqn:reinforce-baseline-unbiased}
   \begin{split}
        \mathbb{E}\_{\tau\sim p\_{\theta}(\tau)}\left[\nabla\_{\theta}\log\pi\_{\theta}(a\_t\mid s\_t) b(s\_t) \right] &=  \mathbb{E}\_{\tau\_{(0:t)}}\left[\mathbb{E}\_{\tau\_{(t+1:T-1)}}\left[ \nabla\_{\theta}\log \pi\_{\theta}(a\_{t}\mid s\_{t})b(s\_{t})\right]\right]  \\\\
        &= \mathbb{E}\_{s\_{0:t}, a\_{0:t-1}}\left[\mathbb{E}\_{s\_{t+1:T}, a\_{t:T-1}}\left[ \nabla\_{\theta}\log \pi\_{\theta}(a\_{t}\mid s\_{t})b(s\_{t})\right]\right] \\\\
        &= \mathbb{E}\_{s\_{0:t}, a\_{0:t-1}}\left[b(s\_{t})\mathbb{E}\_{s\_{t+1:T}, a\_{t:T-1}}\left[ \nabla\_{\theta}\log \pi\_{\theta}(a\_{t}\mid s\_{t})\right]\right] \\\\
        &= \mathbb{E}\_{s\_{0:t}, a\_{0:t-1}}\left[b(s\_{t})\mathbb{E}\_{a\_{t}}\left[ \nabla\_{\theta}\log \pi\_{\theta}(a\_{t}\mid s\_{t})\right]\right] \\\\
        &= \mathbb{E}\_{s\_{0:t}, a\_{0:t-1}}\left[b(s\_{t})\nabla\_{\theta}\mathbb{E}\_{a\_{t}}\left[\log \pi\_{\theta}(a\_{t}\mid s\_{t})\right]\right] \\\\
        &= \mathbb{E}\_{s\_{0:t}, a\_{0:t-1}}\left[b(s\_{t})\nabla\_{\theta}1\right] \\\\
        &= 0.
   \end{split}
\end{equation}
\\]

We can remove irrelevant variables from the expectation over the portion of the trajectory \\(\tau_{(t+1):(T-1)}\\) because we are focusing on the term at step \\(t\\). The only relevant variable is \\(a_{t}\\), and the expectation \\(\mathbb{E}\_{a\_{t}}\log\pi\_{\theta}(a\_{t}\mid s\_{t})\\) is 1. Given that the gradient with respect to \\(\theta\\) of a constant is zero, and \\(b(s\_{t})\\) is multiplying it, the effect of the baseline on the expectation is nullified. This argument can be applied to any other term in the sequence due to the linearity of the expectation. Therefore, we have proven that using a baseline also keeps the gradient estimator unbiased in the policy gradient setting. 

 Choosing an appropriate baseline is a critical decision in reinforcement learning \citep{foundations-deeprl-series-l3}, as different methods can offer unique strengths and limitations. Common baselines include fixed values, moving averages, and learned value functions.

* Constant baseline: \\(b=\mathbb{E}\left[ R(\tau)\right]\approx \frac{1}{m}\sum\_{i=1}^{m} R(\tau^{(i)})\\).
* Optimal constant baseline: \\( b=\frac{\sum\_{i}(\nabla\_{\theta} \log P\_{\theta}(\tau^{(i)}))^{2} R(\tau^{(i)})}{\sum\_{i}(\nabla\_{\theta}\log P\_{\theta}(\tau^{(i)}))^{2}}\\).
* Time-dependent baseline: \\(b\_{t}=\frac{1}{m} \sum\_{i=1}^{m} \sum\_{k=t}^{T-1} R(s\_{k}^{(i)}, a\_{k}^{(i)})\\).
* State-dependent expected return: \\(b(s\_{t}) = \mathbb{E}\left[r\_{t} + r\_{t+1} + r\_{t+2} + \dots + r\_{T-1}\right] = V^{\pi}(s\_{t})\\).


The control variates method can significantly reduce estimator variance, enhancing the stability and performance of RL algorithms \cite{NIPS2001_584b98aa}. Despite the nuances and differences among baseline methods, the primary concept is the _advantage_, shown in Equation~(\ref{eqn:pg-objective-with-value-baseline}), which refers to increase log probabilities of action \\(a_{t}\\) proportionally to how much its returns, \\(r_{t}\\), are better than the expected return under the current policy, which is determined by the value function \\(V^{\pi}(s\_{t})\\)

\\[
\begin{equation}\label{eqn:pg-objective-with-value-baseline}
    \mathbb{E}\_{\tau\sim p\_{\theta}(\tau)} \left[\sum\_{t=0}^{T-1}\nabla\_{\theta}\log\pi\_{\theta}(a\_{t}\mid s\_{t}) \left(\underbrace{\sum\_{t'=t}^{T-1} R(a\_{t'}, s\_{t'}) - V^{\pi}(s\_{t})}\_{\text{advantage}} \right) \right].
\end{equation}
\\]

What remains is how do we get estimates for \\(V^{\pi}\\) in practice.

## Actor-Critic Methods

Actor-Critic referred to learn concurrently models for the policy and the value function. This methods are more data efficient because they amortize the samples collected \\(\mathcal{D}^{\pi\_{\theta}}\\) used for Monte Carlo estimations while reducing the variance of the gradient estimator. The actor controls how the agent behaves---_by updating the policy parameters \\(\theta\\) as we see in previous sections_---whereas the critic measures how good the taken action is, and could be a state-value (\\(V\\)) or action-value (\\(Q\\)) **\footnote**{Action-value function (\\(Q\\)) refers to the value of take action \\(a\\) on state \\(s\\) under a policy \\(\pi\\).} function. Notice that we are combining in some way both approaches for solving MDPs as is depicted in Figure~\ref{fig:rl-model-free-taxonomy}.

 We are introducing a new function approximator for the value function, \\(V_{\phi}(s_{t})\\), where \\(\phi\\) are the parameters of the value function 

\\[
    \begin{equation}\label{eqn:actor-critic-objective}
        \mathbb{E}\_{\tau\sim p\_{\theta}(\tau)} \left[\sum\_{t=0}^{T-1}\nabla\_{\theta}\log\pi\_{\theta}(a\_{t}\mid s\_{t}) \left( \sum\_{t'=t}^{T-1} R(a\_{t'}, s\_{t'}) - V\_{\phi}^{\pi}(s\_{t}) \right) \right].
    \end{equation}
\\]

The objective is to minimize the mean squared error (MSE) between the estimated value and the empirical return, i.e. we are regress the value against empirical return in a supervised learning fashion

<!-- V% \ca{Mencionar conexión con el mse a partir de la varianza del gradiente? (Seita post)}: -->

\\[
    \begin{equation}\label{eqn:value-function-loss}
    \phi \leftarrow \underset{\phi}{\arg\min} \frac{1}{\mid\mathcal{D}^{\pi\_{\theta}}\mid}\sum\_{\tau\in\mathcal{D}^{\pi\_{\theta}}}\sum\_{t=0}^{T-1}\left[\left(\left(\sum\_{t'=t}^{T-1} R(a\_{t'}, s\_{t'})\right) - V\_{\phi}(s\_{t})\right)^2~\right].
    \end{equation}
\\]


[Algorithm 2](#alg:reinforcemet-with-critic) describes the steps for a REINFORCE variant with advantage , which combines the actor-critic approach with the traditioinoal REINFORCE algorithm. More components were introduced and can influence in the performance when the algorithm is implemented. For instance, the policy and value networks can share parameters or not. A useful study that make abalations and suggestions to pay attention when these algorithms are implemented is <i>What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study (Andrychowicz, 2020 \cite{andrychowicz2020mattersonpolicyreinforcementlearning})</i>.


<div id="alg:reinforce-with-critic">
    <big><b>Algorithm 2: REINFORCE with advantage</b></big>
    <ol>
        <li>Initialize policy \( \pi_{\theta} \)</li>
        <li>Initialize value \( V_{\phi} \)</li>
        <li>Set learning rates \( \alpha_{a} \) and \( \alpha_{c} \)</li>
        <li>For \( \text{iteration}=0, 1, 2, \dots, N \):
            <ol>
                <li>Collect a set of trajectories \( \mathcal{D}^{\pi_{\theta}}=\{\tau^{(i)}\} \) by sampling from the current policy \( \pi_{\theta} \)</li>
                <li>Calculate the returns \( R(\tau) \) for each trajectory \( \tau\in\mathcal{D}^{\pi_{\theta}} \)</li>
                <li>Update the policy:
                    <ul>
                        <li>\( \theta \leftarrow \theta + \alpha_{a} \left(\frac{1}{|\mathcal{D}^{\pi_{\theta}}|}\sum_{\tau\in\mathcal{D}^{\pi_{\theta}}}\left[\sum_{t=0}^{T-1}\nabla_{\theta}\log\pi_{\theta}(a_{t}| s_{t})\left(\sum_{t'=t}^{T-1} R(a_{t'}, s_{t'}) - V_{\phi}^{\pi_{\theta}}(s_{t})\right)\right]\right) \)</li>
                    </ul>
                </li>
                <li>Update the value:
                    <ul>
                        <li>\( \phi \leftarrow \phi + \alpha_{c} \left(\frac{1}{|\mathcal{D}^{\pi_{\theta}}|}\sum_{\tau\in\mathcal{D}^{\pi_{\theta}}}\left[\sum_{t=0}^{T-1}\left(\sum_{t'=t}^{T-1} R(a_{t'}, s_{t'}) - V_{\phi}^{\pi_{\theta}}(s_{t})\right)\nabla_{\phi}V_{\phi}^{\pi_{\theta}}(s_{t})\right]\right) \)</li>
                    </ul>
                </li>
            </ol>
        </li>
    </ol>
</div>


## References

<p>
    [1] Sutton, R. S. (2018). 
    <a href="http://incompleteideas.net/book/the-book-2nd.html" target="_blank">Reinforcement learning: An introduction</a>. A Bradford Book.
</p>

<p>
    [2] Mnih, V. (2013). <a href="https://arxiv.org/abs/1312.5602" target="_blank">Playing atari with deep reinforcement learning</a>. arXiv preprint arXiv:1312.5602.
</p>

<p>
    [3] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). <a href="https://www.nature.com/articles/nature24270" target="_blank">Mastering the game of go without human knowledge</a>. nature, 550(7676), 354-359.
</p>

<p>
    [4] Schulman, J. (2016). <a href="https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-217.html" target="_blank">Optimizing expectations: From deep reinforcement learning to stochastic computation graphs</a> (Doctoral dissertation, UC Berkeley).

</p>


<!-- 

## Improving Sample Efficiency: Behavior and Target Policies


The main drawback of the REINFORCE algorithm is its sample complexity. Once we roll out the policy and collect the data, we cannot reuse it after the policy has been updated. We must collect new data following the \textit{target policy} $\pi_{\theta}$ that we want to update. In RL literature, this is referred to as \textit{on-policy} learning. Reusing the data $\mathcal{D}\sim\pi_{\theta_{\text{old}}}$ to update the current policy $\pi_{\theta}$ would significantly improve sample efficiency\footnote{This issue also arises when attempting to transfer behavior from one task to another using existing data.}. However, once we update the policy, the previously collected data is no longer valid because the policy has changed. The distribution from which the data was sampled is now $\pi_{\theta_{\text{old}}}$. \\

 Using behavior data learned from another policy, known as a \textit{behavior policy}, to update the current policy is referred to as \textit{off-policy} learning in RL literature. Let's introduce a \textit{behavior policy} in the RL objective defined in Equation~(\ref{eqn:rl-objective}) using \href{https://timvieira.github.io/blog/post/2014/12/21/importance-sampling/}{importance sampling} (See Mckay book, Section 29.2 \cite{mackay-book}):

% Derive RL objective with importance sampling to use data from another policy
\begin{equation}\label{eqn:derive-rl-objective-with-is}
    \begin{split}
        \nabla_{\theta}\mathcal{J}(\theta) &= \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\bigg[ \nabla_{\theta}\log p_{\theta}(\tau) R(\tau)\bigg] \\
        &= \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \bigg[ \frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta}(\tau)} R(\tau) \bigg] \\
        &= \int_{\mathcal{X}} p_{\theta}(\tau) \frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta}(\tau)} R(\tau) d\tau \\
        &= \int_{\mathcal{X}} \frac{p_{\theta_{\text{old}}}(\tau)}{p_{\theta_{\text{old}}}(\tau)} \cancel{p_{\theta}(\tau)} \frac{\nabla_{\theta}p_{\theta}(\tau)}{\cancel{p_{\theta}}(\tau)} R(\tau) d\tau \\
        &= \int_{\mathcal{X}} p_{\theta_{\text{old}}}(\tau) \frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta_{\text{old}}}(\tau)} R(\tau) d\tau \\
        &= \mathbb{E}_{\tau\sim p_{\theta_{\text{old}}}(\tau)} \bigg[\frac{\nabla_{\theta} p_{\theta}(\tau)}{p_{\theta_{\text{old}}}(\tau)} R(\tau)\bigg].
    \end{split}
\end{equation}

 We derive a new objective that is more general and reconciles both \textit{on-policy} and \textit{off-policy} learning in the importance weight,
or importance correction ($p_{\theta}(\tau) / p_{\theta_{\text{old}}}(\tau)$) 

% RL objective with IS
\begin{equation}\label{eqn:rl-objective-with-is}
    \mathcal{J}_{\text{IS}}(\theta) = \mathbb{E}_{\tau\sim p_{\theta_{\text{old}}}(\tau)}\bigg[\frac{p_{\theta}(\tau)}{p_{\theta_{\text{old}}}(\tau)} R(\tau)\bigg].
\end{equation}

 We can assume that the data collected from the behavior policy is
not so different from the target policy, and use first order approximation to
update the policy 

\begin{equation}\label{eqn:rl-objective-is-linear-aprox}
    \begin{split}
        \nabla_{\theta}\mathcal{J}(\theta)\rvert_{\theta=\theta_{\text{old}}} &= \mathbb{E}_{\tau\sim p_{\theta_{\text{old}}}(\tau)} \bigg[\frac{\nabla_{\theta} p_{\theta}(\tau)\rvert_{\theta=\theta_{\text{old}}}}{p_{\theta_{\text{old}}}(\tau)} R(\tau)\bigg] \\
        &= \mathbb{E}_{\tau\sim p_{\theta_{\text{old}}}(\tau)} \big[\nabla_{\theta}\log p_{\theta}(\tau)\rvert_{\theta=\theta_{\text{old}}} R(\tau) \big].
    \end{split}
\end{equation}


 \textbf{The problem with first order approximation}. The gradient estimation it is good only in the inmediate vecinity, because is a local approximation of the function. Hence, the step size is crucial to avoid a policy degradation, a situation where the policy is updated with a bad gradient,
it is difficult to recover from this situation. Given that the data is collected by the policy, the feedback loop can be dangerous for the training
stability. \\

## Trust Region and Proximal Policy Optimization

Trust Region Policy Optimization (TRPO) \cite{schulman2015trust} allows us to
avoid the policy degradation given bad updates. The idea is to 
define a trust region in which update the policy parameter is safer and
balancing the policy improvement with stability

\begin{align}
    \text{Surrogate loss:} \quad & \underset{\pi_{\theta}}{\max}~L(\pi) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_{\theta}(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right] \label{eqn:trpo-loss} \\
    \text{Constraint:} \quad & \mathbb{E}_{\pi_{\text{old}}} \left[ D_{\text{KL}}(\pi_{\theta} || \pi_{\theta_{\text{old}}}) \right]  \leq \epsilon \nonumber.
\end{align}

% \textbf{Maximize data efficiency in comparison to traditional policy gradients}. 

 Increase data efficiency while avoiding step size problems in updating parameters, compared to traditional policy gradients (PG). The main idea is to improve a surrogate objective significantly while making minimal changes to the policy. These minimal changes are quantified using the KL divergence between action distributions. The trust region is the area where the new policy remains close to the old one, guarantee training stability. \\

% The trust region is the area where the new policy remains close to the old one, allowing for constrained improvement. \\

 Proximal Policy Optimization (PPO) \cite{schulman2017proximal} is about simplify TRPO in order to (i) be easier to implement avoiding solve the second order optimization in Equation~(\ref{eqn:trpo-loss}), (ii) taking advantage of first order optimizer such as ADAM \cite{kingma2017adammethodstochasticoptimization}, and (iii) be more compatible with neural networks operations such as dropout that are incompatible with TRPO setting. \\

 Let's rename the importance weights as the probability ratio $r$: 

\begin{equation}\label{eqn:importance-ratio-is}
    r_{t}(\theta) = \frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}\mid s_{t})}.
\end{equation}

 The strategy is to keep this ratio closer to 1. We can create a trust region via clipping the ratio to force within a range $\left[1-\epsilon, 1+\epsilon \right]$,

\begin{equation}\label{eqn:clip-ac-objective}
\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right].
\end{equation}

 For a walkthrough implementation that cover important details avoid in the paper and that impact significatnly in the performance, review the work \textit{``The 37 implementation details of proximal policy optimization''} (Huang, 2023 \cite{dlr191986}).


% algoritmo naive REINFORCE
\begin{algorithm}
    \caption{Proximal Policy Optimization (PPO), Actor-Critic Style}
    \begin{algorithmic}
    \STATE Initialize policy parameter $\theta$, set learning rate $\alpha$
    \STATE Initialize value $V_{\phi}$
    \FOR{$\text{iteration}=0, 1, 2, \dots N$}
        \FOR{$\text{actor}=0, 1, 2, \dots M$}
            \STATE Run policy $\pi_{\theta_{\text{old}}}$ in environment for $T$ timesteps        
            \STATE Compute advantage estimates $\hat{A}_{0}, \dots, \hat{A}_{T-1}$
        \ENDFOR
        \STATE Optimize surrogate $\mathcal{L}^{\text{CLIP}}$ wrt $\theta$ (Equation~\ref{eqn:clip-ac-objective}), with $K$ epochs and minibatch size $M\leq NT$
    \ENDFOR
    \end{algorithmic}
\end{algorithm}


## Summary

In this chapter, we have explored the foundational concepts and methodologies in reinforcement learning (RL). The core of RL is the interaction between an agent and its environment, where learning occurs through trial-and-error. The agent's goal is to maximize cumulative rewards by taking actions based on its observations, influencing the state of the environment, and receiving rewards. 

 We began by introducing the Markov Decision Process (MDP), a mathematical framework that describes the interaction between the agent and the environment. An MDP is characterized by a state space, action space, transition probabilities, and reward functions. The agent aims to learn a policy that maximizes the expected return, which is the sum of discounted rewards over time. 

 We then delved into policy optimization methods, focusing on policy gradients, a popular approach in model-free RL. Policy gradient methods reduce RL to a problem of stochastic gradient descent, leveraging trajectories of state-action pairs to update the policy parameters. We discussed techniques such as the reward-to-go and baselines to reduce the variance of gradient estimators, thus improving learning efficiency. 

 In conclusion, reinforcement learning offers a powerful framework for designing intelligent agents capable of learning optimal behaviors through interaction with their environment. By understanding and implementing the principles and techniques covered in this chapter, one can develop sophisticated RL agents for a wide range of applications. 
 -->
