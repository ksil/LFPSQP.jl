using Test
using LinearAlgebra
using LFPSQP
using Profile
import ForwardDiff.Dual

# Projected CG tests
include("test_cg.jl")

# Gradient and Hessian generator function tests
include("test_autodiff.jl")