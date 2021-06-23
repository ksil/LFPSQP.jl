function optimize(f, c!, x0::Vector{Float64}, m::Int64, param::LFPSQPParams)

	# generate first-order functions
	grad! = generate_gradient(f, x0)
	jac! = (m == 0) ? nothing : generate_jacobian(c!, x0, m)

	# generate Lagrangian Hessian function
	x0dual = zeros(Dual{nothing,Float64,1}, length(x0))
	x0dual .= x0

	grad_dual! = generate_gradient(f, x0dual)
	jac_dual! = (m == 0) ? nothing : generate_jacobian(c!, x0dual, m)

	hess_lag_vec! = generate_hess_lag_vec(grad_dual!, jac_dual!, x0, m)

	optimize(f, grad!, c!, jac!, hess_lag_vec!, x0, m, param)
end


function optimize(f, grad!, c!, jac!, hess_lag_vec!, x0::Vector{Float64}, m::Int64, param::LFPSQPParams)
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

	# output arrays
	x = copy(x0)
	obj_values = zeros(0)
	n = length(x0)

	# ---------- storage for steps -----------------
	xtilde = similar(x)					# temporary step
	xnew = similar(x)
	Jc = Array{Float64}(undef, m, n) 	# Jacobian of constraints
	Jct = Array{Float64}(undef, n, m)	# gradient of constraints (Jacobian transpose)
	g = Array{Float64}(undef, n)		# gradient of objective function
	d = Array{Float64}(undef, n)		# step to take
	dx = Array{Float64}(undef, n)

	tmp_n = Array{Float64}(undef, n)	# temp vectors
	tmp_m = Array{Float64}(undef, m)
	
	cval = zeros(m)						# to store constraint values
	λ_kkt = zeros(m)
	term_cond::TerminationCondition = f_tol

	# storage and work vectors for SVD
	U = Array{Float64}(undef, n, m)
	S = Vector{Float64}(undef, m)
	Vt = Array{Float64}(undef, m, m)
	svdwork = Vector{Float64}(undef, 1)

	if m > 0
		ksvd!(Jct,U,S,Vt,svdwork,true)		# calculate work vector
	end

	newton_d = Array{Float64}(undef, n)				# truncated Newton steps
	newton_Δλ = Array{Float64}(undef, m)
	newton_b2 = zeros(m)
	projcgwork = ProjCGWork(n, m)
	prev_grad_norm = 0.0					# forces tol to be κ*grad_norm initially
	grad_norm = Inf
	newton_map = LinearMap{Float64}((dest, src) -> hess_lag_vec!(dest, src, x, λ_kkt), n, ismutating=true, issymmetric=true)

	# retraction methods
	nr = NR(U, S, Vt, param.ϵ_c, param.maxiter_retract, NRWork(m))
	pp = ProjPenalty(jac!, U, S, Vt, m, param.μ0, param.ϵ_c, param.maxiter_retract, param.maxiter_pcg, ProjPenaltyWork(m, n))
	euc = Euclidean()

	# linesearch work structs
	armijo_work = ArmijoWork(n)
	exact_work = ExactLinesearchWork(n)

	# ----------------------- Calculate gradient -------------------------
	i = 0
	f_diff = Inf
	step_diff = Inf
	kkt_diff = Inf

	fval = f(x)
	append!(obj_values, fval)

	param.disp == iter && print_iter_header()

	while true
		# calculate gradient at current point
		grad!(g, x)

		# calculate step and add random noise (if set)
		d .= -g
		if param.β > 0 
			randn!(tmp_n)

			if param.t_β > 0
				# if there should be some ramp down of random noise
				axpy!(param.β*max(1 - i/param.t_β, 0.0), tmp_n, d)
			else
				axpy!(param.β, tmp_n, d)
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
				if a < param.ϵ_rank
					rank = j - 1
					break
				end
			end

			# project step onto tangent manifold (using only rank vectors in U)
			kgemv!('T', rank, 1.0, U, d, 0.0, tmp_m)		# tmp_m = U' d
			kgemv!('N', rank, -1.0, U, tmp_m, 1.0, d)		# d = d - U*tmp_m
		end

		kkt_diff = norm(d, Inf)

		# update rank in ProjPenalty
		pp.rank = rank


		steptype = 0
		tn_iter = 0
		tn_res = 0.0

		# calculate λ_kkt
		if m > 0
			# tmp_m contains U'd = -U'g
			@inbounds @simd for j = 1:rank
				tmp_m[j] /= S[j]
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
			Uview = view(U, :, 1:rank)
			newton_b2_view = view(newton_b2, 1:rank)
			
			grad_norm = norm(d)
			tol = param.tn_κ*min(1, (grad_norm/prev_grad_norm)^2)*grad_norm
			
			prev_grad_norm = grad_norm  	# update prev_grad_norm for next time

			# take truncated newton step using ProjCG
			tn_iter, tn_res = projcg!(newton_d, newton_Δλ, newton_map, Uview, d, newton_b2_view,
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
			retract_method = euc
			mtype = 0
		end


		if param.linesearch == armijo || param.disable_linesearch
			# note - don't need to project g onto tangent manifold for dot product since d is guaranteed to live on tangent manifold
			flag, iter1, iter2, newf, f_diff, step_diff, α = armijo!(xnew, x, d, g, f, fval, retract_method, cval, c!, param, armijo_work)
		else
			flag, iter1, iter2, newf, f_diff, step_diff, α = exact_linesearch!(xnew, x, d, f, fval, retract_method, cval, c!, param, exact_work)
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

	return x, obj_values, λ_kkt, TerminationInfo(term_cond, f_diff, step_diff, kkt_diff, i)
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