module LFPSQP

using ForwardDiff
using ForwardDiff: GradientConfig, JacobianConfig, Chunk, Dual, Tag
using ReverseDiff
using LinearAlgebra
using LinearAlgebra.BLAS: gemv!, axpy!, ger!
using Parameters
using Printf
using Random: randn!

# include files
# include("la_helper.jl")
# include("newton_direction.jl")
# include("constrained_descent.jl")
include("projcg.jl")

end # module
