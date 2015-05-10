---
layout: post
title: "Introduction to numerical solutions of population balance equation"
date: 2015-05-10 16:04:58 +0100
comments: true
categories: 
---


##Introduction

[Population balance equation](http://en.wikipedia.org/wiki/Population_balance_equation) (PBE)
allows us to quantify the change of distribution of a single or a set of
descriptors in a sample population. [Seminal work](http://en.wikipedia.org/wiki/Smoluchowski_coagulation_equation) in PBE
was done by [Marian Smoluchowski](http://en.wikipedia.org/wiki/Marian_Smoluchowski), who was a
Polish scientist working on the foundations of statistical physics. A typical
application in fluid dynamics context is a size distribution of a dispersion
such as those encountered in gas-liquid or liquid-liquid flows where bubbles or
drops play the role of the sample population. The methodology is more general
though and has been used in other branches of modern science in order to study
polymerization, biological cells or as models of ecosystems. Also, Lattice
Boltzmann numerical techniques are based on this methodology. We are focused
here on the fluid dynamic application and PBE will be used in order to capture
the change of volume  due to breakup and coalescence processes in bubbles or
drops.

PBE are a set of integro-differential equations derived from Boltzmann equation
for the number density function describing the size. The interaction term
captures the coalescence and breakup processes through integrals over breakup
or coalescence rates and the density function itself. For certain forms of
these kernel functions equations can be solved analytically, but with the
advent of computational methods it is also possible to obtain numerical
approximations to the solutions of kernels of more general type.

In this post I will show a comparison of analytical solutions for pure breakup
and pure coalescence cases. We discretise continuous PBE equations with finite
volume and the choice of internal variable grid follows Hidy and Brock (1970)
paper. The comparisons for pure breakage and agglomeration replicate the PBE
testing reported in paper by Kumar and Ramkrishna (1996).

The exercise is performed in order to develop a calculation tool with simple
python interface. The tests were a side product of other projects that I am
running at the moment.


``` python
    from numpy import arange, zeros, exp, trapz, piecewise, linspace
    from scipy.integrate import odeint
    from itertools import cycle
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    %matplotlib inline
```

## Population balance equations

### Continuous equations

A recent review by Solsvik and Jakobsen (2015) gives three alternative ways of
deriving population balance equations: through continuum dynamics, from
Boltzmann equation and through probabilistic arguments. We begin here by
assuming that these equations are known and that for bubbles or drops we
can formulate them in terms of particle volume. Volume is convenient as it is a
conserved quantity in breakage and coalescence processes. If $v$ is particle
volume and $n(v,t)$ denotes the number density function then the continuous set
of PBE are

$$
\begin{align}
\frac{\partial n(v,t)}{\partial t}
\, = \, &
\frac{1}{2} \int_0^v \! n(v - v', t), n(v',t) Q(v-v', v') \, \mathrm{d} \xi'
\\
& -
\Gamma(v) n(v,t)
\\
& -
\int_0^\infty \! n(v, t) n(v', t) Q(v, v') \, \mathrm{d} v'
\\
& +
\int_v^\infty \beta(v, v') \Gamma(v') n(v', t) \, \mathrm{d} v',
\end{align}
$$

where $Q$ is coalescence rate, $\Gamma$ is the breakage rate and $\beta$ is the
daughter particle distribution.

### Discrete equations

We discretise the equation by applying finite volume method. We will be solving
for a fixed number of equations for quantities:

$$
\begin{equation}
N_i(t) = \int_{v_i}^{v_{i+1}} \! n(v, t)\, \mathrm{d}v
\end{equation}
$$

which represent the total number of drops of size from $[v_i, v_{i+1})$. This
discretisation became knowns as the method of classes (MOC). The tracked
quantites $N_k$ representing the number of particles belonging to a particular
size class. The full discrete equations are then:

$$
\begin{align}
\frac{\partial N_i(t)}{\partial t}
\, = \, &
\frac{1}{2} \int_{v_i}^{v_{i+1}}\int_0^v \! n(v - v', t)\, n(v',t) Q(v-v', v') \, \mathrm{d} v'\mathrm{d} v
\\
& -
\int_{v_i}^{v_{i+1}} \! \Gamma(v) n(v,t) \, \mathrm{d} v
\\
& -
\int_{v_i}^{v_{i+1}}\int_0^\infty \! n(v, t) n(v', t) Q(v, v') \, \mathrm{d} v' \mathrm{d} v
\\
& +
\int_{v_i}^{v_{i+1}} \int_v^\infty \beta(v, v') \Gamma(v') n(v', t) \, \mathrm{d} v'\mathrm{d} v.
\end{align}
$$

At this stage a problem of closure arises. The left hand side represents the
change of known values of our system but the right hand side contains integrals
of function $n$ which is unknown.

Hidy and Brook propose a uniform discretisation of the internal space. We fix
the smallest drop size to $v_1$ and a fixed number of classes. Class $k$
represent then the size $v_k = kv_1$.

We will now turn to the closure problem and applying [mean value
theorem](https://en.wikipedia.org/wiki/Mean_value_theorem#Mean_value_theorems_for_integration])
to the right hand side terms. For simplicity we will do only the breakage terms
i.e. the second and the fourth terms. For the first term it gives us:

\begin{gather}
\int_{v_i}^{v_{i+1}} \! \Gamma(v) n(v,t) \, \mathrm{d} v = \Gamma(\xi) \int_{v_i}^{v_{i+1}} \! n(v,t) \, \mathrm{d} v = \Gamma(\xi) N_i(t)
\end{gather}

for some $\xi \in [v_i, v_{i+1})$. For the fourth term we need to apply mean value theorem for both integrals.

$$
\begin{equation}
\int_{v_i}^{v_{i+1}} \int_v^\infty \beta(v, v') \Gamma(v') n(v', t) \, \mathrm{d} v'\mathrm{d}v
\\
=
\int_{v_i}^{v_{i+1}} \sum_j \int_{v_j}^{v_{j+1}} \beta(v, v') \Gamma(v') n(v', t) \, \mathrm{d} v'\mathrm{d}v
\\
= 
\int_{v_i}^{v_{i+1}} \sum_j \beta(v, \xi_j') \Gamma(\xi_j')  \int_{v_j}^{v_{j+1}}n(v', t) \, \mathrm{d} v'\mathrm{d}v
\\ 
=
\int_{v_i}^{v_{i+1}} \sum_j \beta(v, \xi_j') \Gamma(\xi_j') N_j(t) \, \mathrm{d}v
\\
=
(v_{i+1} - v_i) \sum_j \beta(\xi, \xi_j') \Gamma(\xi_j') N_j(t)
\end{equation}
$$

The unkown $\xi$ and $\xi'$ values could now be selected in order for the equalities to hold. However, we cannot choose these values for arbitrary functions. In here we will somehwat arbitrarily set these unkowns to left sides of each interval and assume that the functions do not vary signficantly. The equalities will only hold approximately, but  the accuracy of approximation can be increased with increased resolution. For pure breakage case the equations are now:

\begin{equation}
\frac{\partial N_i(t)}{\partial t} 
=
-N_i(t) \Gamma_i
+ (v_{i+1} - v_{i}) \sum_{j=i+1}^{M} \beta_{i,j} \Gamma_j N_j
\end{equation}

where the indexed values are simply the values of kernel functions at the left side of each interval. Similar derivation can be made for coalescence. See for instance Kumar and Ramkrishna (1996) paper. You can find the current version of my PBE code in my [pyfd](https://github.com/robertsawko/pyfd) repository. The code is quite straighforward and python `odeint` interface is used to solve the ordinary differential equation that arises after discretisation.

``` python
    class MOCSolution:
        def RHS(
            self, N, t
        ):
            dNdt = zeros(self.number_of_classes)
    
            if self.gamma is not None and self.beta is not None:
                # Death breakup term
                dNdt -= N * self.gamma(self.xi)
                # Birth breakup term
                for i in arange(self.number_of_classes):
                    for j in arange(i + 1, self.number_of_classes):
                        dNdt[i] += \
                            self.beta(self.xi[i], self.xi[j]) \
                            * self.gamma(self.xi[j]) \
                            * N[j] * self.delta_xi
    
            if self.Q is not None:
                for i in arange(self.number_of_classes):
                    # Birth coalescence term
                    for j in arange(0, i):
                        dNdt[i] += 0.5 * N[i - j] * N[j] \
                            * self.Q(self.xi[j], self.xi[i - j])
                    # Death coalescence term
                    for j in arange(self.number_of_classes):
                        dNdt[i] -= N[i] * N[j] * self.Q(self.xi[i], self.xi[j])
            return dNdt
    
        def __init__(self, N0, t, xi0, beta=None, gamma=None, Q=None):
            self.number_of_classes = N0.shape[0]
            # Kernels setup
            self.beta = beta  # Daughter particle distribution
            self.gamma = gamma  # Breakup frequency
            self.Q = Q  #
            # Uniform grid
            self.xi = xi0 + xi0 * arange(self.number_of_classes)
            self.delta_xi = xi0
            # Solve procedure
            self.N = odeint(lambda NN, t: self.RHS(NN, t), N0, t)
```

## Numerical experiments

In this section we will draw some comparisons on log-log plots between
analytical and numerical solutions. I am using `ggplot` style with some minor
alterations. To minimize code duplication I create two functions.
`pbe_solutions` is assumed here to be a dictionary holding grid sizes.

Note that in function `plot_pbe_solutions` in order to reconstruct the number
density function of the continuous formulation you need to divide the values
$N_k$ by the grid size or interval over which original NDF was integrated. You
assume that this ratio is the mean value of the original NDF.

```python
    mpl.style.use('ggplot')
    plt.rcParams.update({
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 16,
        'axes.labelcolor': 'black',
        'ytick.color': 'black',
        'xtick.color': 'black',
        'figure.figsize': [10, 6.12]
    })
    
    
    def plot_total_numbers(pbe_solutions, time, vmax, analytical):
        totals = dict(
            (
                n,
                [sum(Ns) for Ns in pbe_solutions[n].N]
            ) for n in breakup_solutions
        )
        v = linspace(0, vmax, 1000)
        Na = [analytical(t) for t in time]
        fig = plt.figure()
        ax = fig.gca()
        linestyles = cycle(['-', '--', ':'])
        for n in sorted(totals):
            ax.loglog(
                time, totals[n]/totals[n][0],
                linewidth=2,
                linestyle=next(linestyles),
                label="MOC with N={0}".format(n))
        ax.loglog(time, Na / Na[0], "-k", linewidth=4, label="Analytical")
        ax.legend(loc='upper left', shadow=True)
        ax.set_xlabel('Time')
        ax.set_ylabel("Total number of particles $N/N_0$")
        plt.show()
        
        
    def plot_pbe_solutions(pbe_solutions, time, vmax, analytical, initial=None):
        v = linspace(1e-2, vmax, 1000)
        sol = analytical(v, time[-1])
    
        fig = plt.figure()
        ax = fig.gca()
        markers = cycle(['o', 's', 'v', '*', '.', ','])
        for n in sorted(pbe_solutions):
            ax.semilogy(
                pbe_solutions[n].xi, pbe_solutions[n].N[-1] / (vmax/n), "+",
                marker=next(markers),
                label="MOC with N={0}".format(n))
        if initial is not None:
            ax.semilogy(v, initial(v), "--k", linewidth=2, label="Initial condition")
        ax.semilogy(v, sol, "-k", linewidth=2, label="Analytical")
        ax.legend(loc='lower left', shadow=True)
        ax.set_xlabel('Particle volume')
        ax.set_ylabel('Number density function')
        plt.show()
```

## Class convergence for pure breakup

### Analytical solutions

Ziff and Mc Grady (1985) study several different breakup kernels. In here we
focus on their case number 4 which corresponds to a uniform binary breakup.
Their equations after reformulations to our formalism are:

$$
\begin{gather}
\beta(v, v') = \frac{2}{v'},
\\
\Gamma(v) = v^2.
\end{gather}
$$

The uniform aspect is achieved by daughter particle distribution depending only
on the size of a drop which breaks up i.e. $v'$ i.e. it does not depend on the
smaller drop size. Factor of two comes from counting breakage into an ordered
pair $(v, v'-v)$ and $(v'-v, v)$ separately. Alternatively, you can see the
factor as normalisation as $\beta$ has to integrate to the number of drops
produced in the breakage.

The continuous equations then takes the form

$$
\begin{gather}
\frac{\partial n(v,t)}{\partial t} = -v^2 n(v,t)  + 2\int_v^ \infty v' n(v', t) \mathrm{d} v'
\end{gather}
$$

and Ziff and McGrady report the solution to this equation (case 4) for a *monodisperse* configuration i.e. $n(v, 0) = \delta(v - v_0)$ where $v_0$ is fixed. In discrete formulation this will resolve to $N_k(0) = 1$ for $k$ being an interval such that $v_0 \in [v_k, v_{k+1})$.

$$
\begin{equation}
n(v, t) = \left\{
\begin{array}{lr}
2tv_0 e^{-tv^2} & v < v_0
\\
\delta(v - v_0) e^{-tv_0^2} & v = v_0
\\
0 & v > v_0
\end{array}
\right.
\end{equation}
$$

Here's my implementation of this solution with the use of neat `piecewise`
function. I wasn't sure how to represent Dirac's delta in the formulation.

```python
    def zm_pbe_solution(x, t, v0):
        return piecewise(
            x,
            [x < v0, x == v0, x > v0],
            [
                lambda x: 2.0 * t * v0 * exp(-t * x**2),
                lambda x: exp(-v0 * x**2) * 1000,
                lambda x: 0.0
            ]
        )
```
In order to validate the total number of particles we have integrate that solution over all sizes. 

$$
\begin{equation}
N = \int_0^\infty \! n(v, t) \, \mathrm{d} v
=
e^{-tv_0^2} + 
\int_0^\infty 2tv_0 e^{-tv^2} \, \mathrm{d} v
\end{equation}

$$

```python
    def zm_total_number_solution(t, v0, resolution=1000):
        x = linspace(0, v0, resolution)
        return exp(-t * v0**2) \
            + trapz(2.0 * t * v0 * exp(-t * x**2), x=x)
```

#### Numerical solutions
We solve the population balance problem for pure breakup. We assume an
mono-dispersed initial distribution of size $v_0=1$ and we allow it to breakup
for 10s. We take 10, 20, 40, 80 and 160 classes for that setup.

```python
    v0=1.0
    time = arange(0.0, 10.0, 0.001)
    v = 1.0
    grids = [10, 20, 40, 80, 160]
    
    breakup_solutions = dict()
    for g in grids:
        N0 = zeros(g)
        N0[-1] = 1
        breakup_solutions[g] = MOCSolution(
            N0, time, v0 / g,
            beta=lambda x, y: 2.0 / y,
            gamma=lambda x: x**2
        )
```
Finally we plot the results on log, log axes. The total numbers are normalised with respect to initial total number.

```python
    plot_total_numbers(
        breakup_solutions, time, v0,
        lambda t: zm_total_number_solution(t, v0)
    )
```

{% img center /images/intro_pbe-pure_breakup_N.png 'Pure breakup total number solution' %}


A clear convergence to the analytical solution is visible. The figure reveals
that the numerical solutions contain fewer drops then the analytical solutions.
This is due to the fact that the smallest class continues to break up but there
are no smaller classes represented in the discrete form so the drops are
vanishing. In the end all numerical solutions will depart from the analytical
curve due to that lower limit. This behaviour is already visible for $N=10$
case.

```python
    plot_pbe_solutions(
        breakup_solutions, time, v0,
        lambda v, t: zm_pbe_solution(v, t, v0=v0)
    )
```

{% img center /images/intro_pbe-pure_breakup_pbe.png 'Pure breakup number density solution' %}

Again we observe that there's a clear convergence to analytical solution. The
visualisation of the right end of the analytical solution is somewhat
problematic as it's supposed to have a Dirac's delta value. Don't know how to
solve presently.

### Class convergence for pure agglomeration

#### Analytical solutions

Scott (1968) provided a solution for a pure agglomeration case with three
different coalescence kernels: sum, product and constant. I will only do a
constant here although I intend to add the remaining two into my test suite. In
our formulation Scott's case is

$$
\begin{equation}
Q(v, v') = C
\end{equation}
$$

and the remaining kernels are zero.

Also, Scott assumed two types of initial conditions a dual-dispersed case with
a sum of two Dirac's deltas and a Gaussian-like distribution given by

$$
\begin{equation}
n(v, 0) = \frac{N_0}{v_2} \left(\frac{v}{v_2}\right) e^{-v/v_2}
\end{equation}
$$

It is important to observe here - and this observation cost me three hours of
my life - that the mean volume from that distribution is tied to $v_2$ by:


$$
\begin{equation}
v_0 = 2v_2
\end{equation}
$$


```python
    from scipy.special import gamma
    
    def scott_total_number_solution3(t, C=1, N0=1):
        T = C * N0 * t
        return 2.0 * N0 / (T + 2.0)
    
    
    def scott_pbe_solution3(v, t, C=1.0, N0=1.0, v0=1.0):
        T = C * N0 * t
        x = v / v0
        phi3 = sum([
            (x * 2)**(2 * (k + 1)) / gamma(2 * (k + 1)) * (T/(T+2))**k
            for k in range(100)
            ]) * 4.0 * exp(- 2 * x) / (x * (T + 2)**2)
        return N0 / v0 * phi3
```

#### Numerical solutions
Similarly to previous section we set five different grids and a simulation
time. We set the simulation time to be smaller as the particles will keep
increasing its size and therefore can leave our uniform grid if enough time has
passed. `vmax` is set only for visualisation purpose.

```python
    t = arange(0.0, 1, 0.01)
    vmax = 1e1
    v2 = 1.0
    N0 = 1.0
    grids = [10, 20, 40, 80, 160]
    C = 1.0
    
    coalescence_solutions = dict()
    for g in grids:
        dv = vmax / g
        v = dv + dv * arange(g)
        Ninit = (N0 / v2) * (v / v2) * exp(-v / v2) * dv
        coalescence_solutions[g] = MOCSolution(
            Ninit, t, dv,
            Q=lambda x, y: C
        )


    plot_total_numbers(
        coalescence_solutions, t, vmax,
        lambda t: scott_total_number_solution3(t, v2)
    )
```

{% img center /images/intro_pbe-pure_coalescence_N.png 'Pure coalescence total number solution' %}


``` python

    plot_pbe_solutions(
        coalescence_solutions, t, vmax,
        lambda v, t: scott_pbe_solution3(v, t, C=C, N0=N0, v0=2*v2),
        lambda v: N0/v2 * v/v2 * exp(-v/v2)
    )
```


{% img center /images/intro_pbe-pure_coalescence_pbe.png 'Pure coalescence number density solution' %}


## Summary

So far we can see that we have convergence to the analytical solutions. The
number of classes though had to be large in comparison with what we can
feasibly support in a CFD code. If method of classes is to be applied in
two-fluid model it would mean solving an additional $N$ transport equations
where $N$ is the number of classes and this has to be multiplied by a number of
cells which probably limit our applicability to around 40 classes. The would
represent a very accurate but also computationally expensive solution to the
problem of population balance. This is why there is interest in developing
special discretisation strategies or various methods of moments. Book by
Marchiso and Fox reviews many of these other approaches. Method of classes
though could be used as a validation test.


This page is still work in progress. I intend to include:

 * more explanations and context,
 * discussion on the challenges of PBE,
 * simultaneous breakup and coalescence based on Blatz and Tobolsky,
 * moment evolution.
 
The page acts as demonstration of the code that is being developed in [pyfd repository](https://github.com/robertsawko/pyfd).

## References

{% bibliography %}
