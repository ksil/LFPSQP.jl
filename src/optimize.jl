# ---------------------------- Convenience functions ---------------------------------------------

#=
bounds and inequality constraints (d <= 0)

m - size of c
p - size of d

increases the number of variables to (n + p) and the number of equality constraints
to (m + p) via the introduction of slack variables

=#
function optimize(f, c!, d!, x0::Vector{Float64}, xl, xu, m::Int64, p::Int64, param::LFPSQPParams=LFPSQPParams())
	# if no inequality constraints
	if isnothing(d!) || p == 0
		return optimize(f, c!, x0, xl, xu, m, param)
	end

	# fill auxiliary x0 with x0 and the initial values of the slack variables
	n = length(x0)

	x0_aux = similar(x0, n + p)
	x0_aux[1:n] .= x0
	d!(view(x0_aux, n+1:n+p), x0)

	xl_aux = similar(xl, n + p)
	xl_aux[1:n] .= xl
	xl_aux[n+1:n+p] .= -Inf

	xu_aux = similar(xu, n + p)
	xu_aux[1:n] .= xu
	xu_aux[n+1:n+p] .= 0.0

	function f_aux(x)
		return f(view(x, 1:n))
	end

	function c_aux!(cval, x)
		if m > 0
			c!(view(cval, 1:m), view(x, 1:n))
		end

		d!(view(cval, m+1:m+p), view(x, 1:n))
		cval[m+1:m+p] .-= view(x, n+1:n+p)

		return cval
	end

	# generate first-order functions
	grad! = generate_gradient(f_aux, x0_aux)
	jac! = generate_jacobian(c_aux!, x0_aux, m + p)

	# generate Lagrangian Hessian function
	x0_aux_dual = zeros(Dual{nothing,Float64,1}, length(x0_aux))
	x0_aux_dual .= x0_aux

	grad_dual! = generate_gradient(f_aux, x0_aux_dual)
	jac_dual! = generate_jacobian(c_aux!, x0_aux_dual, m + p)

	hess_lag_vec! = generate_hess_lag_vec(grad_dual!, jac_dual!, x0_aux, m + p)

	# truncate output
	x, obj_values, λ_kkt, term_info = optimize(f_aux, grad!, c_aux!, jac!, hess_lag_vec!, x0_aux, xl_aux, xu_aux, m + p, param)
	trunc_x = x[1:n]

	return trunc_x, obj_values, λ_kkt, term_info
end


# bounds and equality constraints
function optimize(f, c!, x0::Vector{Float64}, xl, xu, m::Int64, param::LFPSQPParams=LFPSQPParams())

	# generate first-order functions
	grad! = generate_gradient(f, x0)
	jac! = (m == 0) ? nothing : generate_jacobian(c!, x0, m)

	# generate Lagrangian Hessian function
	x0dual = zeros(Dual{nothing,Float64,1}, length(x0))
	x0dual .= x0

	grad_dual! = generate_gradient(f, x0dual)
	jac_dual! = (m == 0) ? nothing : generate_jacobian(c!, x0dual, m)

	hess_lag_vec! = generate_hess_lag_vec(grad_dual!, jac_dual!, x0, m)

	optimize(f, grad!, c!, jac!, hess_lag_vec!, x0, xl, xu, m, param)
end

# no bounds on x
function optimize(f, c!, x0::Vector{Float64}, m::Int64, param::LFPSQPParams=LFPSQPParams())
	optimize(f, c!, x0, nothing, nothing, m, param)
end



# ------------------------------------------------------------------------------------------------

function optimize(f, grad!, c!, jac!, hess_lag_vec!, x0::Vector{Float64}, xl, xu, m::Int64, param::LFPSQPParams)
	#= perform constrained optimization of the Helfrich energy

	INPUT
	f - objective function
	grad! - gradient of objective function, called as grad!(g, x)
	c! - constraint function, called as c!(cval, x)
	jac! - jacobian of constraint function, called as jac!(Jc, cval, x)
	hess_lag_vec! - function to calculate the action of the Lagrangian Hessian
	x0 - initial guess
	m - number of constraints
	param - algorithmic parameters in a struct LFPSQPParams

	OUTPUT
	x - the point to which the algorithm converged
	obj_values - an array of f evaluated at all steps taken
	kkt_values - an array containing estimates of the kkt multipliers (if they exist)
	term_info - a TerminationInfo struct containing termination criteria

	=#

	n = length(x0)

	# --------------------- check input data --------------------------------

	if !isnothing(xl) && !isnothing(xu)
		if !(length(xl) == length(xu) == length(x0))
			error("xl, xu, and x0 must all be the same length")
		end
	end

	# set up inequality storage and check bounds
	if (isnothing(xl) && isnothing(xu)) || (all(xl .== -Inf) && all(xu .== Inf))
		ineq = false

		f_aug = f

		ineqdata = InequalityData()
	else
		ineq = true

		if any(xl .> xu)
			error("Infeasible: lower bounds cannot be greater than upper bounds")
		end

		f_aug = x -> f(view(x, 1:n))

		PJct = Array{Float64}(undef, 2*n, m)
		λy_kkt = zeros(n)

		ineqdata = InequalityData(xl, xu)
	end

	n_ineq = ineq ? 2*n : n
	m_ineq = ineq ? m+n : m

	# output arrays
	x = Array{Float64}(undef, n_ineq)
	x[1:n] .= x0

	# find initial values for y
	if ineq
		generate_initial_y!(x, ineqdata)
	end

	obj_values = zeros(0)

	# ---------- storage for steps -----------------
	xtilde = similar(x)					# temporary step
	xnew = similar(x)
	Jc = Array{Float64}(undef, m, n) 	# Jacobian of constraints
	Jct = Array{Float64}(undef, n, m)	# gradient of constraints (Jacobian transpose)
	g = zeros(n_ineq)					# gradient of objective function (0 w.r.t. y if inequalities present)
	d = Array{Float64}(undef, n_ineq)	# step to take

	tmp_n = Array{Float64}(undef, n_ineq)	# temp vectors
	tmp_m = Array{Float64}(undef, m_ineq)
	
	cval = zeros(m)						# to store constraint values
	λ_kkt = zeros(m)
	term_cond::TerminationCondition = f_tol

	# storage and work vectors for SVD
	U = Array{Float64}(undef, n_ineq, m)
	Σ = Vector{Float64}(undef, m)
	Vt = Array{Float64}(undef, m, m)
	svdwork = Vector{Float64}(undef, 1)

	if m > 0
		ksvd!(ineq ? PJct : Jct, U, Σ, Vt, svdwork, true)		# calculate work vector
	end

	newton_d = Array{Float64}(undef, n_ineq)				# truncated Newton steps
	newton_Δλ = Array{Float64}(undef, m_ineq)
	newton_b2 = zeros(m_ineq)
	projcgwork = ProjCGWork(n_ineq, m_ineq)
	prev_grad_norm = 0.0					# forces tol to be κ*grad_norm initially
	grad_norm = Inf

	# inequality decomposition
	if ineq
		ineqdecomp = InequalityDecomp(U, Σ, Vt, [Array{Float64}(undef, n) for i in 1:3]..., Jct, m)
	else
		ineqdecomp = InequalityDecomp(U, Σ, Vt, [zeros(0) for i in 1:3]..., Jct, m)
	end
	ineqproject = InequalityDecompProject(ineqdecomp)

	# construct Linear map to be used for Newton steps
	if ineq
		newton_map = LinearMap{Float64}((dest, src) -> augmented_hess_lag_vec!(dest, src, hess_lag_vec!, x, λ_kkt, λy_kkt, ineqdata), 2*n, ismutating=true, issymmetric=true)
	else
		newton_map = LinearMap{Float64}((dest, src) -> hess_lag_vec!(dest, src, x, λ_kkt), n, ismutating=true, issymmetric=true)
	end

	# retraction methods
	nr = NR(U, Σ, Vt, param.ϵ_c, param.maxiter_retract, NRWork(m), ineq, ineqdata)
	pp = ProjPenalty(jac!, U, Σ, Vt, m, param.μ0, param.ϵ_c, param.maxiter_retract, param.maxiter_pcg, ProjPenaltyWork(m, n, m_ineq, n_ineq), ineq, ineqdecomp, ineqdata)
	euc = Euclidean()
	yr = YRetract(ineqdata)

	# linesearch work structs
	armijo_work = ArmijoWork(n_ineq)
	exact_work = ExactLinesearchWork(n_ineq)

	# ----------------------- Calculate gradient -------------------------
	i = 0
	f_diff = Inf
	step_diff = Inf
	kkt_diff = Inf

	fval = f_aug(x)
	append!(obj_values, fval)

	param.disp == iter && print_iter_header()

	while true
		# calculate gradient at current point
		grad!(view(g, 1:n), view(x, 1:n))

		# calculate step and add random noise (if set)
		d .= -1.0 .* g

		if param.β > 0 
			randn!(tmp_n)

			if param.t_β > 0
				# if there should be some ramp down of random noise
				axpy!(param.β*max(1 - i/param.t_β, 0.0), tmp_n, d)
			else
				axpy!(param.β, tmp_n, d)
			end
		end

		if ineq
			# calculate gradients of inequality constraints
			inequality_gradient!(ineqdecomp, x, ineqdata)
		end

		rank = m
		if m > 0
			# calculate Jacobian at current point and find SVD
			jac!(Jc, cval, view(x, 1:n))
			transpose!(Jct, Jc)

			if ineq
				# project Jacobian orthogonal to inequality constraints
				PJct[1:n, :] .= (1.0 .- ineqdecomp.Dx.*ineqdecomp.Dx) .* Jct
				PJct[n+1:2*n, :] .= -1.0 .* ineqdecomp.Dy.*ineqdecomp.Dx .* Jct

				ksvd!(PJct, U, Σ, Vt, svdwork)		# O(Nm^2)
			else
				ksvd!(Jct, U, Σ, Vt, svdwork)		# O(Nm^2)
			end

			# find rank of constraint Jacobian
			for (j, a) in enumerate(Σ)
				if a < param.ϵ_rank
					rank = j - 1
					break
				end
			end

			# project step onto tangent space if no inequalities present
			if !ineq
				kgemv!('T', rank, 1.0, U, d, 0.0, tmp_m)		# tmp_m = U' d
				kgemv!('N', rank, -1.0, U, tmp_m, 1.0, d)		# d = d - U*tmp_m
			end
		end

		# project step onto tangent space for inequalities
		if ineq
			# update rank in inequality decomposition
			ineqdecomp.rank = rank

			mul!(tmp_m, ineqproject', d)				# tmp_m = Q'd
			mul!(d, ineqproject, tmp_m, -1.0, 1.0)		# d = d - Q*tmp_m
		end

		kkt_diff = norm(d, Inf)

		# update rank in ProjPenalty
		pp.rank = rank


		steptype = 0
		tn_iter = 0
		tn_res = 0.0

		# calculate λ_kkt
		if ineq
			calculate_λ_kkt!(λ_kkt, λy_kkt, tmp_m, ineqdecomp)
		elseif m > 0
			# tmp_m contains U'd = -U'g
			@inbounds @simd for j = 1:rank
				tmp_m[j] /= Σ[j]
			end
			@inbounds @simd for j = rank+1:m
				tmp_m[j] = 0.0
			end

			gemv!('T', 1.0, Vt, tmp_m, 0.0, λ_kkt) # λ = -V Σ^{-1} U' g
		end

		# -------------------- Check for termination conditions --------------

		if f_diff <= param.ϵ_f
			term_cond = f_tol
			break
		elseif step_diff <= param.ϵ_x
			term_cond = x_tol
			break
		elseif i >= param.maxiter
			term_cond = max_iter
			break
		elseif kkt_diff <= param.ϵ_kkt
			term_cond = kkt_tol
			break
		end


		# -------------------- Truncated Newton step -------------------------

		if param.do_newton
			# prepare views
			if ineq
				Qview = ineqproject
				newton_b2_view = view(newton_b2, 1:n+rank)
			else
				Qview = view(U, :, 1:rank)
				newton_b2_view = view(newton_b2, 1:rank)
			end
			
			# calculate tolerance for truncated newton step
			grad_norm = norm(d)
			tol = param.tn_κ*min(1, (grad_norm/prev_grad_norm)^2)*grad_norm
			
			prev_grad_norm = grad_norm  	# update prev_grad_norm for next time

			# take truncated newton step using ProjCG
			tn_iter, tn_res = projcg!(newton_d, newton_Δλ, newton_map, Qview, d, newton_b2_view,
				tol=tol, maxit=param.tn_maxiter, work=projcgwork)


			# choose Newton direction if in gradient direction
			if dot(newton_d, d) > 0.0
				d .= newton_d
				steptype = 1
			end
		end


		# ---------------------- Linesearch ------------------------

		# determine retraction method to use
		if m > 0
			if rank == m && !param.do_project_retract
				retract_method = nr
				mtype = 0
			else
				retract_method = pp
				mtype = 1
			end
		else
			if ineq
				retract_method = yr
				mtype = 0
			else
				retract_method = euc
				mtype = 0
			end
		end


		if param.linesearch == armijo || param.disable_linesearch
			# note - don't need to project g onto tangent manifold for dot product since d is guaranteed to live on tangent manifold
			flag, iter1, iter2, newf, f_diff, step_diff, α = armijo!(xnew, x, n, d, g, f_aug, fval, retract_method, cval, c!, param, armijo_work)
		else
			flag, iter1, iter2, newf, f_diff, step_diff, α = exact_linesearch!(xnew, x, n, d, f_aug, fval, retract_method, cval, c!, param, exact_work)
		end

		
		# ----------------------- update x and function values ---------------------------
		x .= xnew
		fval = newf
		append!(obj_values, fval)
		param.disp == iter && print_iter(i+1, fval, norm(cval,Inf), f_diff, step_diff, steptype, tn_iter, tn_res, mtype, iter1, iter2, α, flag)
		
		i += 1

		# callback
		if param.callback != nothing && mod(i, param.callback_period) == 0
			param.callback(i, x)
		end
	end


	if i == param.maxiter
		@warn "Maximum # of outer iterations reached"
	end

	return x[1:n], obj_values, λ_kkt, TerminationInfo(term_cond, f_diff, step_diff, kkt_diff, i)
end

@inline function print_iter_header()
	@printf("   step |          f     ||c||      |Δf|    ||Δx||  |   S iter      res  |   M   iter  (pcg)  |        α  flag\n")
	@printf("--------------------------------------------------------------------------------------------------------------\n")
end

@inline function print_iter(i, fval, normc, fstep, normx, steptype, tn_iter, tn_res, methodtype, iter1, iter2, α, flag)
	# print out iteration information
	if methodtype == 0
		method = "NR"
	else
		method = "PP"
	end

	if steptype == 0
		stepname = "GD"
	else
		stepname = "TN"
	end

	@printf("%7d | %10.3e  %8.1e  %8.1e  %8.1e  |  %s %4d %8.1e  |  %s %6d %6d  | %8.1e  %4d\n",
		i, fval, normc, fstep, normx, stepname, tn_iter, tn_res, method, iter1, iter2, α, flag)

	flush(Base.stdout)
end