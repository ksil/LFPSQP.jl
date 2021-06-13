using IterativeSolvers
using IterativeSolvers: gmres_iterable!, GMRESIterable
using LinearMaps
using ForwardDiff
using ForwardDiff: Dual
using LinearAlgebra
using LinearAlgebra.BLAS: gemv!

struct NewtonWork
	duals_n::Vector{Dual{nothing,Float64,1}}
	duals_jac::Array{Dual{nothing,Float64,1}, 2}
	x_dual::Vector{Dual{nothing,Float64,1}}
	rank::Array{Int64,1}
	prev_grad::Array{Float64,1}
	mat_v_prod!::Function
	L::LinearMap{Float64}
	iterable::GMRESIterable
	κ::Float64
end

function NewtonWork(n::Int64, m::Int64, x::Vector{Float64}, λ::Vector{Float64}, dual_f!, dual_c!, U::Array{Float64, 2}, gmres_maxiter::Int64=n+m, κ::Float64=1.0)
	#= constructs the mapping and GMRES iterable object to be used
	Note that y cannot be updated so this vector must remain the same when changing its values

	INPUT
	n - # of variables
	m - # of constraints
	x - n-long vector containing current x
	λ - m-long vector containing current λ
	dual_f! - function to calculate gradient of f on dual numbers
	dual_c! - function to calculate jacobian of c on dual numbers
	U - SVD matrix of the constraint Jacobian
	gmres_maxiter - max number of gmres iterations
	κ - parameter to establish gmres convergence

	OUTPUT
	a NewtonWork struct
	=#

	duals_n = zeros(Dual{nothing,Float64,1}, n)
	duals_jac = zeros(Dual{nothing,Float64,1}, m, n)
	x_dual = zeros(Dual{nothing,Float64,1}, n)
	rank = [m]
	# # fdiff_res = zeros(m+n)
	# # fdiff_tmp = zeros(m+n)

	# x2 = y[1:n]

	# for j = 1:1
	# 	g1 = zeros(n)
	# 	g2 = zeros(n)
	# 	v = randn(n)
	# 	@show h = 1e-8*norm(x2)/norm(v)

	# 	fd_hv = zeros(n)
	# 	ad_hv = zeros(n)

	# 	grad!(g1, x2 .+ h*v)
	# 	grad!(g2, x2)

	# 	fd_hv .= (g1 .- g2) / (h)

	# 	x_dual[1:n] .= x2 .+ Dual{nothing}(0.0, 1.0)*v
	# 	x_dual[n+1:m] .= 0.0
	# 	dual_grad!(duals, x_dual)
	# 	@inbounds @simd for i = 1:n
	# 		ad_hv[i] = duals[i].partials[1]
	# 	end

	# 	# fdiff_fxn = (q, a) -> dual_grad!(q, [x2; zeros(m)] .+ a*[v; zeros(m)])
	# 	# ForwardDiff.derivative!(fdiff_res, fdiff_fxn, fdiff_tmp, 0.0)
	# 	# ad_hv[1:n] .= fdiff_res[1:n]

	# 	@show norm(fd_hv .- ad_hv, Inf)
	# 	@show fd_hv[1:min(n,10)]
	# 	@show ad_hv[1:min(n,10)]
	# 	@show norm(fd_hv .- ad_hv, Inf) ./ norm(ad_hv, Inf)
	# 	@show sum(isnan.(ad_hv))

	# end

	# -------------- define functions and linear map -------------------

	function mat_v_prod!(dest, src)
		@inbounds src_x = view(src, 1:n)
		@inbounds src_l = view(src, n+1:n+m)
		@inbounds dest_x = view(dest, 1:n)
		@inbounds dest_l = view(dest, n+1:n+m)

		a = Dual{nothing}(0.0, 1.0)
		x_dual .= x .+ a*src_x

		# ----- first block -------
		dual_f!(duals_n, x_dual)

		# extract Hessian-vector products
		@inbounds @simd for i = 1:n
			dest[i] = duals_n[i].partials[1]
		end

		if rank[1] == 0
			return dest
		end

		dual_c!(duals_jac, x_dual)

		@inbounds for j = 1:n
			@simd for i = 1:m
				dest[j] -= duals_jac[i,j].partials[1]*λ[i]	# negative since we need to use -λ, where λ = V Σ^{-1} U' g
			end
		end

		# ----- second block -------
		kgemv!('N', rank[1], 1.0, U, src_l, 1.0, dest_x)

		# ----- third block --------
		kgemv!('T', rank[1], 1.0, U, src_x, 0.0, dest_l)

		# ----- fourth block (if present) -------
		@inbounds @simd for i = n+rank[1]+1:m
			dest[i] = src[i]
		end

		return dest
	end
	
	L = LinearMap{Float64}(mat_v_prod!, m+n, ismutating=true, issymmetric=true)

	iterable = gmres_iterable!(zeros(m+n), L, zeros(m+n); maxiter=gmres_maxiter, restart=min(gmres_maxiter, m+n), initially_zero=true)

	return NewtonWork(duals_n, duals_jac, x_dual, rank, [0.0], mat_v_prod!, L, iterable, κ)
end


function newton_direction!(d::Vector{Float64}, b::Vector{Float64}, rank::Int64, nw::NewtonWork)
	#= performs truncated Newton to generate a step
	Note that it is NOT guaranteed that the resulting step lies in the tangent manifold, so
	a projection step after this is necessary
	
	INPUT
	d - m+n vector of Newton step
	b - m+n RHS vector containing [-grad_f; 0]
	rank - rank of constraint Jacobian
	nw - instance of a NewtonWork struct

	OUTPUT
	d - overwritten with Newton step
	duals - overwritten
	=#

	# -------------------- GMRES iteration --------------------------
	it = nw.iterable

	# update initial guess with zeros
	fill!(d, 0.0)
	it.x = d

	# update right-hand side and tolerance
	it.b = b
	norm_b = norm(b)
	it.reltol = nw.κ*min((norm_b/nw.prev_grad[1])^1.2, norm_b)
	nw.prev_grad[1] = norm_b

	# update rank
	nw.rank[1] = rank

	# initialize first residual
	it.β = IterativeSolvers.init!(it.arnoldi, it.x, it.b, it.Pl, it.Ax, initially_zero=true)
    IterativeSolvers.init_residual!(it.residual, it.β)
    it.mv_products = 1
    it.k = 1
    it.residual.current = it.β
	
	# do iteration
	gmres_iters = 0
	res = 0.0
	for (i, residual) = enumerate(it)
		gmres_iters = i
		res = residual
	end

	# returns the number of gmres iterations and the residual
	return gmres_iters, res
end