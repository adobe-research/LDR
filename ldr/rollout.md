# LDR rollout

## 1. Paper's logic (Eq. 2 / Alg. 1)

$$
\begin{aligned}
\ddot{\boldsymbol{s}}_{t-2}&=\ddot{\boldsymbol{s}}_{t-3}+\Delta t\cdot\dddot{\boldsymbol{s}}_{t-3}\approx\ddot{\boldsymbol{s}}_{t-3}+f_\theta(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})\\
\dot{\boldsymbol{s}}_{t-1}&=\dot{\boldsymbol{s}}_{t-2}+\Delta t\cdot\ddot{\boldsymbol{s}}_{t-2}\\
\boldsymbol{s}_{t}&=\boldsymbol{s}_{t-1}+\Delta t\cdot\dot{\boldsymbol{s}}_{t-1}\\
\end{aligned}
$$

## 2. Code implementation

Instead of using $\ddot{\boldsymbol{s}}_{t-3}+f_\theta(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})$, we implement a logically equivalent version: $\textcolor{red}{\ddot{\boldsymbol{s}}_{0}}+f_\theta(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})$

## 3. Why they are equivalent

Given $t\in\{3, 4, \cdots, T\}$, we derive:

$$
\begin{aligned}
\ddot{\boldsymbol{s}}_{t-2}&\approx\ddot{\boldsymbol{s}}_{t-3}+f_\theta(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})\\
&=(\ddot{\boldsymbol{s}}_{t-4}+f_\theta(\dot{\boldsymbol{s}}_{t-4}, \boldsymbol{s}_{t-4}))+f_\theta(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})\\
&\ \ \vdots\\
&=\ddot{\boldsymbol{s}}_{0}+\sum_{k=0}^{t-3}f_\theta(\dot{\boldsymbol{s}}_{k}, \boldsymbol{s}_{k})\\
&\equiv\ddot{\boldsymbol{s}}_{0}+f_\theta(\dot{\boldsymbol{s}}_{0},\dot{\boldsymbol{s}}_{1},\cdots,\dot{\boldsymbol{s}}_{t-3},\boldsymbol{s}_{0},\boldsymbol{s}_{1},\cdots,\boldsymbol{s}_{t-3})
\end{aligned}
$$

where $\equiv$ is "logical equivalence".

Since the rollout is a deterministic recurrence from a fixed initial state, the current state $(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})$ is a sufficient statistic for the whole trajectory (the Markov property): it determines every earlier state $(\dot{\boldsymbol{s}}_{k}, \boldsymbol{s}_{k})$ with $0\le k<t-3$. Thus, the earlier arguments are redundant and can be dropped:

$$
\begin{aligned}
f_\theta(\dot{\boldsymbol{s}}_{0},\dot{\boldsymbol{s}}_{1},\cdots,\dot{\boldsymbol{s}}_{t-3},\boldsymbol{s}_{0},\boldsymbol{s}_{1},\cdots,\boldsymbol{s}_{t-3})&\equiv
f_\theta(\dot{\boldsymbol{s}}_{1},\dot{\boldsymbol{s}}_{2},\cdots,\dot{\boldsymbol{s}}_{t-3},\boldsymbol{s}_{1},\boldsymbol{s}_{2},\cdots,\boldsymbol{s}_{t-3})\\
&\equiv
f_\theta(\dot{\boldsymbol{s}}_{2},\dot{\boldsymbol{s}}_{3},\cdots,\dot{\boldsymbol{s}}_{t-3},\boldsymbol{s}_{2},\boldsymbol{s}_{3},\cdots,\boldsymbol{s}_{t-3})\\
&\ \ \vdots\\
&\equiv
f_\theta(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})
\end{aligned}
$$

This yields $\ddot{\boldsymbol{s}}_{0}+f_\theta(\dot{\boldsymbol{s}}_{t-3}, \boldsymbol{s}_{t-3})$, exactly the code's implementation.
