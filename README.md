# LFPSQP.jl

Julia implementation of **L**ocally **F**easibly **P**rojected **S**equential **Q**uadratic **P**rogramming

Performs feasible nonlinear constrained optimization (i.e., every iterate satisfies the proposed constraints) on problems of the form:

![optimization_problem](https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Brl%7D%20%5Cdisplaystyle%20%5Cmin_%7B%5Cmathbf%20x%20%5Cin%20%5Cmathbb%20R%5En%7D%20%26%20f%28%5Cmathbf%20x%29%20%5C%5C%20%5Cmathrm%7Bs.t.%7D%20%26%20%5Cmathbf%20c%28%5Cmathbf%20x%29%20%3D%20%5Cmathbf%200%20%5C%5C%20%26%20%5Cmathbf%20d%5El%20%5Cleq%20%5Cmathbf%20d%28%5Cmathbf%20x%29%20%5Cleq%20%5Cmathbf%20d%5Eu%20%5C%5C%20%26%20%5Cmathbf%20x%5El%20%5Cleq%20%5Cmathbf%20x%20%5Cleq%20%5Cmathbf%20x%5Eu%20%5Cend%7Barray%7D)

where _f_ is the objective function, **c** are equality constraints, **d** are inequality constraints with lower and upper bounds of **d**<sup>l</sup> and **d**<sup>u</sup>, and **x**<sup>l</sup> and **x**<sup>u</sup> are box constraints.

Please cite as:  
K.S. Silmore and J.W. Swan, Locally Feasibly Projected Sequential Quadratic Programming for Nonlinear Programming on Arbitrary Smooth Constraint Manifolds, [ArXiv:2111.03236 \[math.OC\]](https://arxiv.org/abs/2111.03236) (2021).

# Examples
## Unconstrained
Let's optimize the classic Rosenbrock function:
```julia
f = x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2

x0 = [0.0, 0.0]

x, obj_values, λ_kkt, term_info = optimize(f, x0)
```

Here, ``x`` is the final value of the variables, ``obj_values`` is a vector of objective function values at each iterate (including the initial one), ``λ_kkt`` is a vector of Lagrange multipliers (here an empty vector), and ``term_info`` is a struct that summarizes the optimization procedure. For example,
```julia
term_info
```
produces
```
TerminationInfo:
condition = f_tol
       Δf = 1.0898882046786806e-7
   ||Δx|| = 0.0007384068067118611
||P(∇f)|| = 4.332627751789361e-5
    iters = 17
```
where ``condition`` is the termination criterion (here the change in objective function value was smaller than the default tolerance), ``Δf`` is the change in objective function value at the last iteration, ``||Δx||`` is the 2-norm of increment of the variables at the last iteration, ``||P(∇f)||`` is the 2-norm of the final objective function gradient, and ``iters`` is the total number of (outer) iterations taken.

## Equality constrained
```julia
n = 50     # number of variables
m = 1      # number of constraints

f = x -> dot(x, x)

function c!(cval, x)
    cval[1] = x[1] - 0.75
end

x0 = ones(n)

x, obj_values, λ_kkt, term_info = optimize(f, c!, x0, m)
```

## Inequality constrained
```julia
n = 50     # number of variables
m = 0      # number of equality constraints
p = 1      # number of inequality constraints

coeff = randn(n)
f = x -> dot(coeff, x)

# circle constraint
function d!(dval, x)
    dval[1] = dot(x, x) - 1.0
end

x0 = zeros(n)

xl = -Inf .* ones(n)       # no bound constraints
xu = Inf .* ones(n)

x, obj_values, λ_kkt, term_info = optimize(f, nothing, d!, x0, xl, xu, m, p)
```

# To-do
- [ ] Documentation
- [ ] Handling of sparsity in constraints
- [ ] Other miscellaneous features
