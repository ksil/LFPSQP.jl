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

# parameters
import Base.show

# redifine tags to avoid errors
# NOTE that this may break nested differentation, but I don't believe this can be the case
# based on the way that I use these dual types in newton_direction.jl
@inline function ForwardDiff.:≺(::Type{Tag{F1,V1}}, Nothing) where {F1,V1}
    return false
end

@inline function ForwardDiff.:≺(Nothing, ::Type{Tag{F1,V1}}) where {F1,V1}
    return true
end

@enum DisplayOption begin
    off
    iter
end

@enum LinesearchOption begin
    armijo
    exact
end

@enum TerminationCondition begin
	f_tol
	x_tol
	kkt_tol
	max_iter
	armijo_error
end

struct TerminationInfo
	condition::TerminationCondition
	f_diff::Float64
	step_diff::Float64
	kkt_diff::Float64
	iter::Int64
end

Base.show(io::IO, ti::TerminationInfo) = print(io, "TerminationInfo:\ncondition = $(ti.condition)\n" * 
	"       Δf = $(ti.f_diff)\n   ||Δx|| = $(ti.step_diff)\n||P(∇f)|| = $(ti.kkt_diff)\n    iters = $(ti.iter)")


@with_kw struct LFPSQPParams
	α::Float64 = 1.0
	β::Float64 = 0.0
	t_β::Int64 = 0
	s::Float64 = 0.5
	σ::Float64 = 1e-4
	ϵ_c::Float64 = 1e-6
	ϵ_f::Float64 = 1e-6
	ϵ_x::Float64 = 0.0
	ϵ_kkt::Float64 = 1e-6
	ϵ_rank::Float64 = 1e-10
	maxiter::Int64 = 10000
	maxiter_retract::Int64 = 100
	maxiter_pcg::Int64 = 100
	μ0::Float64 = 1e-2
	disable_linesearch::Bool = false
	do_project_retract::Bool = false
	disp::DisplayOption = iter
	callback::Union{Nothing,Function} = nothing
	callback_period::Int64 = 100
	linesearch::LinesearchOption = armijo
	do_newton::Bool = true
	tn_maxiter::Int64 = 10000
	tn_κ::Float64 = 0.5
end

# include files
include("projcg.jl")
include("autodiff_generators.jl")
include("la_helper.jl")
include("inequality_helper.jl")

include("linesearch.jl")
include("retractions.jl")

include("optimize.jl")


export LFPSQPParams
export optimize

end # module
