using Test
using LinearAlgebra
using LFPSQP
import ForwardDiff.Dual

# projected CG tests
include("test_cg.jl")

# gradient and Hessian generator function tests
include("test_autodiff.jl")

# test retraction validity (for equality constraints)
include("test_retractions.jl")

# test linesearches
include("test_linesearch.jl")

# test structs and associated functions to handle inequality constraints
include("test_inequalities.jl")