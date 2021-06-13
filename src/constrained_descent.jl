using ForwardDiff
using ForwardDiff: GradientConfig, JacobianConfig, Chunk, Dual, Tag
using ReverseDiff
using LinearAlgebra
using LinearAlgebra.BLAS: gemv!, axpy!, ger!
using Parameters
using Printf
using Random: randn!

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

@with_kw struct DescentParams
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
	maxiter_nr::Int64 = 100
	maxiter_pcg::Int64 = 100
	λ0::Float64 = 1e-2
	disable_linesearch::Bool = false
	disp::DisplayOption = iter
	callback::Union{Nothing,Function} = nothing
	callback_period::Int64 = 100
	linesearch::LinesearchOption = armijo
	armijo_period::Int64 = 100
	do_newton::Bool = true
	gmres_maxiter::Int64 = 10000
	gmres_κ::Float64 = 0.5
end

function constrained_descent(f, f_grad, c!, c_grad!, x0::Vector{Float64}, m::Int64, p::DescentParams)
	#= does constrained descent using reverse AD, but uses the functions *_for_grad
	if, say, f and c! need to be rewritten to not include external storage
	=#

	# gradient
	cfg_f = ReverseDiff.GradientConfig(x0)
	tape_f = ReverseDiff.GradientTape(f_grad, x0, cfg_f)
	ctape_f = ReverseDiff.compile(tape_f)

	function grad!(g, x)
		ReverseDiff.gradient!(g, ctape_f, x)
	end

	if m > 0 
		# jacobian
		tmp_m = zeros(m)
		cfg_c = ReverseDiff.JacobianConfig(tmp_m, x0)
		tape_c = ReverseDiff.JacobianTape(c_grad!, tmp_m, x0, cfg_c)
		ctape_c = ReverseDiff.compile(tape_c)

		function jac!(Jc, cval, x)
			ReverseDiff.jacobian!(Jc, ctape_c, x)
			c!(cval, x)
		end
	else
		jac! = nothing
	end

	if p.do_newton
		dual_f!, dual_c! = generate_dual_grad(f_grad, c_grad!, x0, m)
	else
		dual_f! = nothing
		dual_c! = nothing
	end

	constrained_descent(f, grad!, c!, jac!, dual_f!, dual_c!, x0, m, p::DescentParams)
end

function constrained_descent(f, c!, x0::Vector{Float64}, m::Int64, p::DescentParams)
	constrained_descent(f, f, c!, c!, x0::Vector{Float64}, m::Int64, p::DescentParams)
end

function constrained_descent_forward(f, c!, x0, m, p::DescentParams, ::Val{CHUNKSIZE}=Val(min(8,length(x0)))) where CHUNKSIZE
	cfg_f = GradientConfig(nothing, x0, Chunk{CHUNKSIZE}())
	grad!(g, x) = ForwardDiff.gradient!(g, f, x, cfg_f)

	if m > 0
		cfg_c = JacobianConfig(nothing, zeros(m), x0, Chunk{CHUNKSIZE}())
		jac!(Jc, cval, x) = ForwardDiff.jacobian!(Jc, c!, cval, x, cfg_c)
	else
		jac! = nothing
	end

	if p.do_newton
		dual_f!, dual_c! = generate_dual_grad(f_grad, c_grad!, x0, m)
	else
		dual_f! = nothing
		dual_c! = nothing
	end

	constrained_descent(f, grad!, c!, jac!, dual_f!, dual_c!, x0, m, p::DescentParams)
end

function generate_dual_grad(f, c!, x0, m)
	#= generates the function that calulcates Hessian vector products for Newton iteration

	INPUTS
	f - objective function f(x) -> val
	c! - constraint function c!(cval, x)
	x0 - the initial x condition
	m - number of constraints

	OUTPUT
	dual_f! - calculates the gradient of f on dual inputs
	dual_c! - calculates the jacobian of c on dual inputs

	=#

	n = length(x0)

	tmp_duals_n = zeros(Dual{nothing,Float64,1}, n)
	tmp_duals_n .= x0
	tmp_duals_m = zeros(Dual{nothing,Float64,1}, m)

	cfg_f = ReverseDiff.GradientConfig(tmp_duals_n)
	tape_f = ReverseDiff.GradientTape(f, tmp_duals_n, cfg_f)
	ctape_f = ReverseDiff.compile(tape_f)

	function dual_f!(dest, src)
		ReverseDiff.gradient!(dest, ctape_f, src)
	end

	if m > 0
		cfg_c = ReverseDiff.JacobianConfig(tmp_duals_m, tmp_duals_n)
		tape_c = ReverseDiff.JacobianTape(c!, tmp_duals_m, tmp_duals_n, cfg_c)
		ctape_c = ReverseDiff.compile(tape_c)

		function dual_c!(dest, src)
			ReverseDiff.jacobian!(dest, ctape_c, src)
		end
	else
		dual_c! = nothing
	end

	return dual_f!, dual_c!
end


function constrained_descent(f, grad!, c!, jac!, dual_f!, dual_c!, x0::Vector{Float64}, m::Int64, p::DescentParams)
	#= perform constrained optimization of the Helfrich energy

	INPUT
	f - objective function
	grad! - gradient of objective function, called as grad!(g, x)
	c! - constraint function, called as c!(cval, x)
	jac! - jacobian of constraint function, called as jac!(Jc, cval, x)
	dual_f! - function to compute Hessian vector product of f
	dual_c! - function to compute Hessian vector product of the constraints, c
	x0 - initial guess
	m - number of constraints
	p - algorithmic parameters in a struct DescentParams
	CHUNKSIZE - chunk size to use during automatic differentiation

	OUTPUT
	x - the point to which the algorithm converged
	obj_values - an array of f evaluated at all steps taken
	kkt_values - an array containing estimates of the kkt multipliers (if they exist)
	term_info - a TerminationInfo struct containing termination criteria

	=#

	# output arrays
	x = copy(x0)
	obj_values = zeros(0)
	n = length(x0)

	# ---------- set up storage -----------------
	xtilde = similar(x)					# temporary step
	xnew = similar(x)
	Jc = Array{Float64}(undef, m, n) 	# Jacobian of constraints
	Jct = Array{Float64}(undef, n, m)	# gradient of constraints (Jacobian transpose)
	D = Array{Float64}(undef, m, m)
	g = Array{Float64}(undef, n)		# gradient of objective function
	d = Array{Float64}(undef, n)		# step to take
	newton_sol = Array{Float64}(undef, n+m)
	@inbounds newton_sol_d = view(newton_sol, 1:n)
	newton_b = zeros(n+m)
	dx = Array{Float64}(undef, n)
	tmp_n = Array{Float64}(undef, n)	# temp vectors
	tmp_n2 = Array{Float64}(undef, n)
	tmp_n3 = Array{Float64}(undef, n)
	tmp_n4 = Array{Float64}(undef, n)	# used by exact linesearch
	tmp_n5 = Array{Float64}(undef, n)
	tmp_n6 = Array{Float64}(undef, n)
	tmp_m = Array{Float64}(undef, m)	# temp vector to do projection
	tmp_m2 = Array{Float64}(undef, m)
	tmp_m3 = Array{Float64}(undef, m)
	cval = zeros(m)						# to store constraint values
	λ_kkt = zeros(m)
	term_cond::TerminationCondition = f_tol

	# storage and work vectors for SVD
	U = Array{Float64}(undef, n, m)
	S = Vector{Float64}(undef, m)
	Vt = Array{Float64}(undef, m, m)
	svdwork = Vector{Float64}(undef, 1)

	# work struct for truncated Newton steps
	nw = NewtonWork(n, m, x, λ_kkt, dual_f!, dual_c!, U, p.gmres_maxiter, p.gmres_κ)

	if m > 0
		ksvd!(Jct,U,S,Vt,svdwork,true)		# calculate work vector
	end

	# ----------------------- Calculate gradient -------------------------
	i = 0
	f_diff = Inf
	step_diff = Inf
	kkt_diff = Inf
	α = p.α

	failed_newton = 0
	can_do_newton_after = 0

	fval = f(x)
	append!(obj_values, fval)

	p.disp == iter && print_iter_header()

	while true
		# calculate gradient at current point
		grad!(g, x)

		# calculate step and add random noise (if set)
		d .= -g
		if p.β > 0 
			randn!(tmp_n)

			if p.t_β > 0
				# if there should be some ramp down of random noise
				axpy!(p.β*max(1 - i/p.t_β, 0.0), tmp_n, d)
			else
				axpy!(p.β, tmp_n, d)
			end
		end

		rank = m
		if m > 0
			# calculate Jacobian at current point and find SVD
			jac!(Jc, cval, x)
			transpose!(Jct, Jc)

			ksvd!(Jct,U,S,Vt,svdwork)		# O(Nm^2)

			# find rank of constraint Jacobian
			for (j, a) in enumerate(S)
				if a < p.ϵ_rank
					rank = j - 1
					break
				end
			end

			# project step onto tangent manifold (using only rank vectors in U)
			kgemv!('T', rank, 1.0, U, d, 0.0, tmp_m)		# tmp_m = U' d
			kgemv!('N', rank, -1.0, U, tmp_m, 1.0, d)		# d = d - U*tmp_m
		end

		kkt_diff = norm(d, Inf)


		steptype = 0
		gmres_iter = 0
		gmres_res = 0.0

		# calculate λ_kkt
		if m > 0
			# tmp_m contains U'd = -U'g
			@inbounds @simd for j = 1:rank
				tmp_m[j] /= S[j]
			end
			@inbounds @simd for j = rank+1:m
				tmp_m[j] = 0.0
			end

			gemv!('T', -1.0, Vt, tmp_m, 0.0, λ_kkt) # λ = V Σ^{-1} U' g
		end

		# -------------------- Check for termination conditions --------------

		if f_diff <= p.ϵ_f
			term_cond = f_tol
			break
		elseif step_diff <= p.ϵ_x
			term_cond = x_tol
			break
		elseif i >= p.maxiter
			term_cond = max_iter
			break
		elseif kkt_diff <= p.ϵ_kkt
			term_cond = kkt_tol
			break
		end


		# -------------------- Truncated Newton step -------------------------

		if p.do_newton && i >= can_do_newton_after
			# take truncated Newton step
			newton_b[1:n] .= d # the rest of the entries are 0.0
			gmres_iter, gmres_res = newton_direction!(newton_sol, newton_b, rank, nw::NewtonWork)

			# ensure step is on tangent manifold
			kgemv!('T', rank, 1.0, U, newton_sol_d, 0.0, tmp_m)		# tmp_m = U' d
			kgemv!('N', rank, -1.0, U, tmp_m, 1.0, newton_sol_d)		# d = d - U*tmp_m

			# choose Newton direction if gradient-related
			if dot(newton_sol_d, d) > 1e-8
				d .= newton_sol_d
				steptype = 1
			else
				failed_newton += 1
			end

			if failed_newton == 20
				failed_newton = 0
				can_do_newton_after = i + 1000
			end
		end


		#####################################################################################################
		gmres_res = kkt_diff

		newf = fval
		flag = 0
		nr_iter = 0
		pb_iter = 0
		pcg_iter = 0
		mtype = 0

		if p.linesearch == exact
			@goto EXACT_LINESEARCH
		end

		# ---------- Armijo line search --------------

		while f_diff > p.ϵ_f && step_diff > p.ϵ_x
			dx .= α.*d
			xtilde .= x .+ dx

			if m > 0
				if rank == m
					# full rank, so do Newton Raphson
					flag, nr_iter = NR!(c!, cval, xtilde, xnew, U, S, Vt, D, tmp_m, tmp_m2, tmp_m3, p.ϵ_c, p.maxiter_nr)
					mtype = 0
				else
					# not full rank, so do primal penalty projection
					flag, pb_iter, pcg_iter = project_penalty!(c!, cval, xtilde, xnew, U, S, Vt, rank, Jc, tmp_m, tmp_m2, tmp_n, tmp_n2, tmp_n3, dx, p)
					mtype = 1
				end

				if flag > 0
					p.disp == iter && print_iter(i+1, fval, norm(cval,Inf), 0.0, 0.0, steptype, gmres_iter, gmres_res, mtype, nr_iter, pb_iter, pcg_iter, α, flag)
					α *= p.s
					continue
				end
			else
				xnew .= xtilde
			end

			# calculate new function values
			tmp_n .= xnew .- x
			newf = f(xnew)

			step_diff = norm(tmp_n)
			f_diff = abs(newf - fval)

			# break conditions
			if p.disable_linesearch
				break
			end

			if (newf - fval) <= p.σ * dot(g, tmp_n)
				break
			end

			p.disp == iter && print_iter(i+1, fval, norm(cval,Inf), 0.0, 0.0, steptype, gmres_iter, gmres_res, mtype, nr_iter, pb_iter, pcg_iter, α, flag)

			α *= p.s

			# to prevent infinite loop
			if α < 1e-100
				@error("Armijo line search failed")
				return x, obj_values, λ_kkt, TerminationInfo(armijo_error, f_diff, step_diff, kkt_diff, i)
			end

			flag = 0
			nr_iter = 0
			pb_iter = 0
			pcg_iter = 0
		end

		# try to increase Armijo step
		if p.armijo_period > 0 && mod(i, p.armijo_period) == 0 && !p.disable_linesearch
			α /= p.s
		end

		@goto END_LINESEARCH


		# ---------- Exact line search --------------

		@label EXACT_LINESEARCH
		ϕ1 = (3 - sqrt(5))/2
		ϕ2 = (sqrt(5) - 1)/2
		ϕ3 = (sqrt(5) + 1)/2
		Δ = α     		# use previous α as step length guess
		f_a = 0.0
		f_b = 0.0
		f_c = 0.0
		f_d = 0.0
		α_a = 0.0
		α_b = 0.0
		α_c = 0.0
		α_d = 0.0
		x_a = xtilde
		x_b = tmp_n4
		x_c = tmp_n5
		x_d = tmp_n6
		do_shrinking = true
		exact_iter_count = 0

		# find the upper bound and maintain rotating list of points
		x_d .= x
		f_d = fval

		# println("Growing")
		while true
			exact_iter_count += 1
			# @printf("xs: %f %f %f %f\n", x_a[1], x_b[1], x_c[1], x_d[1])
			# @printf("αs: %f %f %f %f\n", α_a[1], α_b[1], α_c[1], α_d[1])
			# @printf("fs: %f %f %f %f\n", f_a[1], f_b[1], f_c[1], f_d[1])
			# println("---")

			swp = x_b	# rotate all of the position vectors
			x_b = x_c
			x_c = x_d
			x_d = swp

			f_b = f_c
			f_c = f_d
			α_b = α_c
			α_c = α_d

			dx .= (α_d + Δ).*d
			x_d .= x .+ dx

			if m > 0
				if rank == m
					# full rank, so do Newton Raphson
					flag, nr_iter = NR!(c!, cval, x_d, xnew, U, S, Vt, D, tmp_m, tmp_m2, tmp_m3, p.ϵ_c, p.maxiter_nr)
					mtype = 0
				else
					# not full rank, so do primal penalty projection
					flag, pb_iter, pcg_iter = project_penalty!(c!, cval, x_d, xnew, U, S, Vt, rank, Jc, tmp_m, tmp_m2, tmp_n, tmp_n2, tmp_n3, dx, p)
					mtype = 1
				end

				x_d .= xnew 	# update x_d with projected step
			end

			α_d += Δ

			# should break if projection failed or if α > 1.0
			if flag > 0 || α_d > 1.0
				f_d = Inf 		# Inf indicates projection failed
				break
			end

			f_d = f(x_d)

			if f_d > f_c
				break
			end

			do_shrinking = false
			Δ *= ϕ3
		end

		# no feasible upper bound found, so perform shrinking 
		if do_shrinking
			# println("Shrinking")
			# assign b with α=0 so that point a is assigned correctly below
			f_b = fval
			α_b = 0.0
			x_b .= x

			f_c = Inf
			α_c = Δ

			swp = x_d	# swap with d since d contains the point for α=Δ already
			x_d = x_c
			x_c = swp

			while true
				exact_iter_count += 1
				# @printf("xs: %f %f %f %f\n", x_a[1], x_b[1], x_c[1], x_d[1])
				# @printf("αs: %f %f %f %f\n", α_a[1], α_b[1], α_c[1], α_d[1])
				# @printf("fs: %f %f %f %f\n", f_a[1], f_b[1], f_c[1], f_d[1])
				# println("---")

				swp = x_d	# swapping is now "up" toward d since Δ is getting shrunk
				x_d = x_c
				x_c = swp

				f_d = f_c
				α_d = α_c

				dx .= (ϕ1*α_c).*d
				x_c .= x .+ dx

				if m > 0
					if rank == m
						# full rank, so do Newton Raphson
						flag, nr_iter = NR!(c!, cval, x_c, xnew, U, S, Vt, D, tmp_m, tmp_m2, tmp_m3, p.ϵ_c, p.maxiter_nr)
						mtype = 0
					else
						# not full rank, so do primal penalty projection
						flag, pb_iter, pcg_iter = project_penalty!(c!, cval, x_c, xnew, U, S, Vt, rank, Jc, tmp_m, tmp_m2, tmp_n, tmp_n2, tmp_n3, dx, p)
						mtype = 1
					end

					x_c .= xnew 	# update x_c with projected step
				end

				α_c *= ϕ1

				# should break if projection failed or if α > 1.0
				if flag > 0 || α_c > 1.0
					f_c = Inf
				else
					f_c = f(x_c)
				end

				if f_c <= fval || α_c < 1e-100
					break
				end
			end
		end

		# assign values from upper bounding procedure and calculate point c
		exact_iter_count += 1
		f_a = f_b
		f_b = f_c
		α_a = α_b
		α_b = α_c

		swp = x_a
		x_a = x_b
		x_b = x_c
		x_c = swp

		α_c = α_a + ϕ2*(α_d - α_a)
		dx .= α_c.*d
		x_c .= x .+ dx

		if m > 0
			if rank == m
				# full rank, so do Newton Raphson
				flag, nr_iter = NR!(c!, cval, x_c, xnew, U, S, Vt, D, tmp_m, tmp_m2, tmp_m3, p.ϵ_c, p.maxiter_nr)
				mtype = 0
			else
				# not full rank, so do primal penalty projection
				flag, pb_iter, pcg_iter = project_penalty!(c!, cval, x_c, xnew, U, S, Vt, rank, Jc, tmp_m, tmp_m2, tmp_n, tmp_n2, tmp_n3, dx, p)
				mtype = 1
			end

			x_c .= xnew 	# update x_d with projected step
		end

		if flag > 0 || α_c > 1.0
			f_c = Inf
		else
			f_c = f(x_c)
		end

		# println("Going into loop")

		# do golden ratio bisection
		nd = norm(d)
		while (α_c - α_b) > 1e-6/nd
			exact_iter_count += 1
			# @printf("%f %f %f %f\n", x_a[1], x_b[1], x_c[1], x_d[1])
			# @printf("αs: %f %f %f %f\n", α_a[1], α_b[1], α_c[1], α_d[1])
			# @printf("fs: %f %f %f %f\n", f_a[1], f_b[1], f_c[1], f_d[1])
			# println("---")

			if f_b < f_c || isinf(f_c)	# shrink interval to the left
				swp = x_d
				x_d = x_c
				x_c = x_b
				x_b = swp
				
				f_d = f_c
				f_c = f_b
				α_d = α_c
				α_c = α_b

				# calculate point b (which can never be infinite)
				α_b = α_a + ϕ1*(α_d - α_a)
				dx .= α_b.*d
				x_b .= x .+ dx

				if m > 0
					if rank == m
						# full rank, so do Newton Raphson
						flag, nr_iter = NR!(c!, cval, x_b, xnew, U, S, Vt, D, tmp_m, tmp_m2, tmp_m3, p.ϵ_c, p.maxiter_nr)
						mtype = 0
					else
						# not full rank, so do primal penalty projection
						flag, pb_iter, pcg_iter = project_penalty!(c!, cval, x_b, xnew, U, S, Vt, rank, Jc, tmp_m, tmp_m2, tmp_n, tmp_n2, tmp_n3, dx, p)
						mtype = 1
					end

					x_b .= xnew 	# update x_d with projected step
				end

				f_b = f(x_b)
			else 						# shrink interval to the right
				swp = x_a
				x_a = x_b
				x_b = x_c
				x_c = swp
				
				f_a = f_b
				f_b = f_c
				α_a = α_b
				α_b = α_c

				# calcaulte point c
				α_c = α_a + ϕ2*(α_d - α_a)
				dx .= α_c.*d
				x_c .= x .+ dx

				if m > 0
					if rank == m
						# full rank, so do Newton Raphson
						flag, nr_iter = NR!(c!, cval, x_c, xnew, U, S, Vt, D, tmp_m, tmp_m2, tmp_m3, p.ϵ_c, p.maxiter_nr)
						mtype = 0
					else
						# not full rank, so do primal penalty projection
						flag, pb_iter, pcg_iter = project_penalty!(c!, cval, x_c, xnew, U, S, Vt, rank, Jc, tmp_m, tmp_m2, tmp_n, tmp_n2, tmp_n3, dx, p)
						mtype = 1
					end

					x_c .= xnew 	# update x_d with projected step
				end

				if flag > 0 || α_c > 1.0
					f_c = Inf
				else
					f_c = f(x_c)
				end
			end
		end

		# println("Final values")
		# @printf("xs: %f %f %f %f\n", x_a[1], x_b[1], x_c[1], x_d[1])
		# @printf("αs: %f %f %f %f\n", α_a[1], α_b[1], α_c[1], α_d[1])
		# @printf("fs: %f %f %f %f\n", f_a[1], f_b[1], f_c[1], f_d[1])

		# assign final function, x, and α value
		if f_b < f_c
			xnew .= x_b
			newf = f_b
			α = α_b
		else
			xnew .= x_c
			newf = f_c
			α = α_c
		end

		tmp_n .= xnew .- x
		step_diff = norm(tmp_n)
		f_diff = abs(newf - fval)
		flag = exact_iter_count

		@label END_LINESEARCH

		# update x and function values
		x .= xnew
		fval = newf
		append!(obj_values, fval)
		p.disp == iter && print_iter(i+1, fval, norm(cval,Inf), f_diff, step_diff, steptype, gmres_iter, gmres_res, mtype, nr_iter, pb_iter, pcg_iter, α, flag)
		
		i += 1

		# callback
		if p.callback != nothing && mod(i, p.callback_period) == 0
			p.callback(i, x)
		end
	end


	if i == p.maxiter
		@warn "Maximum # of outer iterations reached"
	end

	return x, obj_values, λ_kkt, TerminationInfo(term_cond, f_diff, step_diff, kkt_diff, i)
end

@inline function print_iter_header()
	@printf("   step |          f     ||c||      |Δf|    ||Δx||  |   S iter      res  |   M   iter  (pcg)  |        α  flag\n")
	@printf("--------------------------------------------------------------------------------------------------------------\n")
end

@inline function print_iter(i, fval, normc, fstep, normx, steptype, gmres_iter, gmres_res, methodtype, nr_iter, pb_iter, pcg_iter, α, flag)
	# print out iteration information
	if methodtype == 0
		method = "NR"
		iter = nr_iter
	else
		method = "PP"
		iter = pb_iter
	end

	if steptype == 0
		stepname = "GD"
	else
		stepname = "TN"
	end

	@printf("%7d | %10.3e  %8.1e  %8.1e  %8.1e  |  %s %4d %8.1e  |  %s %6d %6d  | %8.1e  %4d\n",
		i, fval, normc, fstep, normx, stepname, gmres_iter, gmres_res, method, iter, pcg_iter, α, flag)

	flush(Base.stdout)
end

# ====================== HELPER FXNS ================================

function NR!(c!, cval, xtilde, xnew, U, S, Vt, D, tmp_m, tmp_m2, dc, tol, maxiter)
	#= performs Newton Raphson for c(xtilde + U d) using the Jacobian V Σ'
	which is evaluated at xtilde

	INPUT
	c! - constraint function of the form c!(cval, x) where y is overwritten
	cval - contraint value vector
	xtilde - current x
	xnew - vector in which to store the result
	U, S, Vt - SVD decomposition
	D - m x m array to store inverse of Jacobian of c(xtilde + Ud)
	tmp_m, tmp_m2, dc - work vector of size m
	tol - function tolerance to obtain
	maxiter - maximum number of allowable iterations

	OUTPUT
	cval, xnew, tmp_m overwritten
	flag - 0 = success, 1 = maxiter reached

	=#

	m = length(S)
	c!(cval, xtilde)
	xnew .= xtilde

	# calculate inverse Jacobian
	for j in 1:m
		@fastmath @inbounds @simd for k in 1:m
			D[k,j] = Vt[k,j] / S[k]
		end
	end

	i = 0
	while i < maxiter
		# check if tolerance met
		if norm(cval, Inf) < tol
			break
		end

		# take Newton-Raphson step - tmp_m stores step
		# gemv!('N', -1.0, Vt, cval, 0.0, tmp_m)	# d = - (V Σ') \ cval = - Σ^(-1) V' cval
		# tmp_m ./= S
		gemv!('N', -1.0, D, cval, 0.0, tmp_m)
		gemv!('N', 1.0, U, tmp_m, 1.0, xnew)	# xnew = xnew + U d

		# update function values
		c!(tmp_m2, xnew)

		# Broyden update
		dc .= tmp_m2 .- cval
		cval .= tmp_m2

	    gemv!('T', 1.0, D, tmp_m, 0.0, tmp_m2) # D' dx
	    gemv!('N', -1.0, D, dc, 1.0, tmp_m)

	    alpha = 1 / dot(tmp_m2, dc)
	    ger!(alpha, tmp_m, tmp_m2, D)

		i += 1
	end

	flag = 0
	if i == maxiter
		flag = 1
	end

	return flag, i
end

function pcg!(λ, J, U, S, Vt, rank, x, r, u, c, tmp_m, tol, maxiter)
	#= performs preconditioned conjugate gradient to solve Ax = b with
		A = J' J + λI
		preconditioned with (J0' J0 + λΙ)^(-1) = 1/λ Ι - 1/λ UΣ(λI + Σ^2)^(-1) Σ' U'
		
		Adapted from IterativeSolvers.jl

		INPUT
		λ - factor for projection
		J - the Jacobian of the constraints
		U, S, Vt - SVD decomposition of J(x_i), used for preconditioning
		rank - SVD rank
		x - initial guess
		r - residual, should equal b (the right-hand side) using an initial guess of x=0
		u, c - work vectors of size n for PCG
		tmp_m - work vector of size m
		tol - tolerance for convergence
		maxiter - maximum number of iterations allowed

		OUTPUT
		x, r, u, c, tmp_m overwritten
		x - overwritten with solution
		r - overwritten with residual
		flag - 0 = success, 1 = maxiter reached
	=#

	m = length(S)

	residual = Inf
	ρ = 1.0
	fill!(u, 0.0)

	# calculate residual - r = b - Ax
	# r .= x
	# gemv!('N', 1.0, J, r, 0.0, tmp_m)
	# gemv!('T', -1.0, J, tmp_m, -λ, r)
	# axpy!(1.0, b, r)

	i = 0
	while residual > tol && i < maxiter
		# precondition
		# ldiv!(c, Pl, r)
		c .= r
		kgemv!('T', rank, 1.0, U, r, 0.0, tmp_m)
		@fastmath @inbounds @simd for j in 1:rank
			tmp_m[j] *= S[j]*S[j] / (λ + S[j]*S[j])
		end
		kgemv!('N', rank, -1/λ, U, tmp_m, 1/λ, c)

		ρ_prev = ρ
		ρ = dot(c, r)

		# u := c + βu (almost an axpy)
		β = ρ / ρ_prev
		u .= c .+ β .* u

		# c = A * u
		c .= u
		gemv!('N', 1.0, J, u, 0.0, tmp_m)
		gemv!('T', 1.0, J, tmp_m, λ, c)
		α = ρ / dot(u, c)

		# Improve solution and residual
		axpy!(α, u, x)
		axpy!(-α, c, r)

		residual = norm(r)

		i += 1
	end

	flag = 0
	if i == maxiter
		flag = 1
	end

	return flag, i
end

function project_penalty!(c!, cval, xtilde, xnew, U, S, Vt, rank, J, tmp_m, dc, r, u, c, dx, p)
	#= performs primal penalty minimization of 
		1/2 || c(z) ||^2 + λ/2 || z - xtilde ||^2
	as λ → 0
	assumes gradient is given by Jct c + λ (z - xtilde)
	assumes the Hessian is of the form Jct Jc + λ I = U Σ^2 U' + λ I

	INPUT
	c! - constraint function of the form c!(cval, x) where y is overwritten
	cval - contraint value vector
	xtilde - current x
	xnew - vector in which to store the result
	U, S, Vt - SVD decomposition
	rank - SVD rank
	J - m x n Jacobian at x_i
	dc, tmp_m - work vectors of size m
	r, u, c, dx - work vectors of size n (ASSUMES dx contains xtilde - x_i)
	p - parameters for convergence

	OUTPUT
	cval, xnew, tmp_m, dc, r, u, c, dx overwritten
	flag - 0 = success, 1 = maxiter reached, 2 = pcg maxiter reached

	=#

	flag = 0 # return flag

	# initialize x vectors and calculate constraint functions
	xnew .= xtilde
	λ = p.λ0
	c!(tmp_m, xnew)

	# do initial Broyden update for J(xtilde) - assumes c(x_i) = 0.0 and dx = xtilde - x_i
	dc .= tmp_m
    gemv!('N', -1.0, J, dx, 1.0, dc)

	alpha = 1 / dot(dx, dx)
	ger!(alpha, dc, dx, J)

	cval .= tmp_m

	i = 0
	pcg_iter_count = 0
	while i < p.maxiter_nr
		# check if tolerance met
		if norm(cval, Inf) < p.ϵ_c
			break
		end

	    # cval = c(xnew);

	    # calculate right-hand side r = (J' c + λ*(xnew - xtilde))
	    r .= xnew .- xtilde
	    gemv!('T', 1.0, J, cval, λ, r)
	    fill!(dx, 0.0)

	    # do Newton step = (J'J + λI) \ r
	    pcgf, pcgi = pcg!(λ, J, U, S, Vt, rank, dx, r, u, c, tmp_m, p.ϵ_c, p.maxiter_pcg)
	    pcg_iter_count += pcgi
	    if pcgf > 0
	    	# do no further iterations if pcg exceeds the maximum iteration count
	    	flag = 2
	    	break
	    end
	    xnew .-= dx

	    # calculate new constraint function values
	    c!(tmp_m, xnew)

	    # Broyden update - J = J + (dc - J*dx)*dx'/sum(dx.*dx);
	    dx .*= -1.0 # need to flip sign
	    dc .= tmp_m .- cval
	    gemv!('N', -1.0, J, dx, 1.0, dc)

	    alpha = 1 / dot(dx, dx)
	    ger!(alpha, dc, dx, J)

	    # update count, λ, and function values
		i += 1
		λ *= 0.1
		cval .= tmp_m
	end

	if i == p.maxiter_nr
		flag = 1
	end

	return flag, i, pcg_iter_count

end