module LFPSQP

using ForwardDiff
using ForwardDiff: GradientConfig, JacobianConfig, Chunk, Dual, Tag
using ReverseDiff
using LinearAlgebra
using LinearAlgebra.BLAS: gemv!, axpy!, ger!
using Parameters
using Printf
using Random: randn!
using LinearMaps

# include files
include("projcg.jl")
include("autodiff_generators.jl")
include("la_helper.jl")

include("retractions.jl")
include("optimize.jl")


export LFPSQPParams
export optimize

end # module
