struct NRWork
	D::Array{Float64, 2}
	tmp_m::Vector{Float64}
	tmp_m2::Vector{Float64}
	dc::Vector{Float64}
end

NRWork(m::Int) = NRWork(Array{Float64}(undef, m, m), [Array{Float64}(undef, m) for i in 1:3]...)

mutable struct NR
	U::Array{Float64, 2}
	S::Array{Float64, 1}
	Vt::Array{Float64, 2}
	tol::Float64
	maxiter::Int
	work::NRWork
end

struct ProjPenaltyWork
	J::Array{Float64, 2}
	tmp_m::Vector{Float64}
	r::Vector{Float64}
	p::Vector{Float64}
	z::Vector{Float64}
	dx::Vector{Float64}
	g::Vector{Float64}
end

ProjPenaltyWork(m::Int, n::Int) = ProjPenaltyWork(Array{Float64}(undef, m, n), 
	Array{Float64}(undef, m), [Array{Float64}(undef, n) for i in 1:5]...)

mutable struct ProjPenalty{F <: Function}
	jac!::F
	U::Array{Float64, 2}
	S::Array{Float64, 1}
	Vt::Array{Float64, 2}
	rank::Int
	μ0::Float64
	tol::Float64
	maxiter::Int
	maxiter_pcg::Int
	work::ProjPenaltyWork
end

struct Euclidean
end

function retract!(cval, xnew, c!, xtilde, method::Euclidean)
	xnew .= xtilde

	return 0, 0, 0
end


function retract!(cval, xnew, c!, xtilde, method::NR)
	#= performs Newton-Raphson retraction for c(xtilde + U d) using the Jacobian V Σ'
	which is evaluated at x

	termination criterion is tolerance in inf norm of c!

	INPUT
	cval - contraint value vector
	xnew - vector in which to store the result
	c! - constraint function of the form c!(cval, x) where y is overwritten
	xtilde - current value of x at step in tangent space
	U, S, Vt - SVD decomposition
	tol - function tolerance for termination
	maxiter - maximum number of allowable iterations
	work - NRWork struct for storage

	OUTPUT
	flag - 0 = success, 1 = maxiter reached
	i - total number of iterations taken

	OVERWRITTEN
	cval, xnew

	=#

	# unpack
	U = method.U
	S = method.S
	Vt = method.Vt
	tol = method.tol
	maxiter = method.maxiter
	work = method.work

	# extract quantities from NRWork
	D = work.D    			# m x m array to store inverse of Jacobian of c(xtilde + Ud)
	tmp_m = work.tmp_m
	tmp_m2 = work.tmp_m2
	dc = work.dc

	m = length(S)
	c!(cval, xtilde)
	xnew .= xtilde

	# calculate inverse Jacobian Σ^(-1) V'
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
		gemv!('N', -1.0, D, cval, 0.0, tmp_m)	# d = - (V Σ') \ cval = - Σ^(-1) V' cval
		gemv!('N', 1.0, U, tmp_m, 1.0, xnew)	# xnew = xnew + U d

		# update function values
		c!(tmp_m2, xnew)

		# Broyden update
		dc .= tmp_m2 .- cval
		cval .= tmp_m2

		# Good Broyden
		gemv!('T', 1.0, D, tmp_m, 0.0, tmp_m2) # D' dx
		gemv!('N', -1.0, D, dc, 1.0, tmp_m)		# tmp_m = Δx - D*Δc

	    alpha = 1 / dot(tmp_m2, dc)
	    ger!(alpha, tmp_m, tmp_m2, D)

	    # # Bad Broyden
	    # gemv!('N', -1.0, D, dc, 1.0, tmp_m)		# tmp_m = Δx - D*Δc

	    # alpha = 1 / dot(dc, dc)
	    # ger!(alpha, tmp_m, dc, D)

		i += 1
	end

	flag = 0
	if i == maxiter
		flag = 1
	end

	return flag, i, 0
end

function pcg!(μ, J, U, S, Vt, rank, x, r, p, z, tmp_m, tol, maxiter)
	#= performs preconditioned conjugate gradient to solve Ax = b with
		A = J' J + μI
		preconditioned with (J0' J0 + μΙ)^(-1) = 1/μ Ι - 1/μ UΣ(μI + Σ^2)^(-1) Σ' U'

		INPUT
		μ - factor for projection
		J - the Jacobian of the constraints
		U, S, Vt - SVD decomposition of J(x_i), used for preconditioning
		rank - SVD rank
		x - initial guess
		r - residual, should equal b (the right-hand side) using an initial guess of x=0
		p, z - work vectors of size n for PCG
		tmp_m - work vector of size m
		tol - tolerance for convergence
		maxiter - maximum number of iterations allowed

		OUTPUT
		x, r, p, z, tmp_m overwritten
		x - overwritten with solution
		r - overwritten with residual
		flag - 0 = success, 1 = maxiter reached
	=#

	m = length(S)

	norm_res = Inf
	ρ = 1.0               	# set ρ and p initialy to 1.0 and 0.0, as done in IterativeSolvers.jl
	fill!(p, 0.0)

	i = 0
	while norm_res > tol && i < maxiter
		# precondition - z = M^{-1} r
		z .= r
		kgemv!('T', rank, 1.0, U, r, 0.0, tmp_m)
		@fastmath @inbounds @simd for j in 1:rank
			tmp_m[j] *= S[j]*S[j] / (μ + S[j]*S[j])
		end
		kgemv!('N', rank, -1/μ, U, tmp_m, 1/μ, z)

		# update ρ = r^T z
		ρ_prev = ρ
		ρ = dot(z, r)

		# update β and direction p
		β = ρ / ρ_prev
		p .= z .+ β.*p

		# store A*p in z
		z .= p
		gemv!('N', 1.0, J, p, 0.0, tmp_m)
		gemv!('T', 1.0, J, tmp_m, μ, z)

		# update α = r^T z / (p^T A p)
		α = ρ / dot(p, z)

		# update solution
		# x = x + α*p
		# r = r - α*A*p
		axpy!(α, p, x)
		axpy!(-α, z, r)

		norm_res = norm(r)

		i += 1
	end

	flag = 0
	if i == maxiter
		flag = 1
	end

	return flag, i
end

function retract!(cval, xnew, c!, xtilde, method::ProjPenalty{F}) where F

	#= performs primal penalty minimization of 
		1/2 || c(z) ||^2 + μ/2 || z - xtilde ||^2
	as μ → 0
	
	gradient is given by Jct c + μ (z - xtilde)
	uses Gauss-Newton Hessian of the form Jct Jc + μ I = U Σ^2 U' + μ I

	INPUT
	cval - contraint value vector
	xnew - vector in which to store the result
	c! - constraint function of the form c!(cval, x) where y is overwritten
	xtilde - current x
	jac! - function to calculate Jacobian of the form jac!(J, cval, x)
	U, S, Vt - SVD decomposition of Jacobian at xtilde
	rank - SVD rank
	μ0 - initial penalty strength μ0
	tol - function tolerance for termination
	maxiter - maximum number of (outer) iterations
	maxiter_pcg - maximum number of (inner) pcg iterations
	work - ProjPenaltyWork struct for storage

	OUTPUT
	flag - 0 = success, 1 = maxiter reached, 2 = pcg maxiter reached, 3 = linesearch failed
	i - total number of (outer) iterations taken
	pcg_iter_count - total cumulative number of (inner) pcg iterations taken

	OVERWRITTEN
	cval, xnew

	=#

	# unpack
	jac! = method.jac!
	U = method.U
	S = method.S
	Vt = method.Vt
	rank = method.rank
	μ0 = method.μ0
	tol = method.tol
	maxiter = method.maxiter
	maxiter_pcg = method.maxiter_pcg
	work = method.work

	# unpack work variables
	J = work.J
	tmp_m = work.tmp_m
	r = work.r
	p = work.p
	z = work.z
	dx = work.dx
	g = work.g


	flag = 0 # return flag

	# initialize x vectors and calculate constraint functions
	xnew .= xtilde
	μ = μ0


	i = 0
	pcg_iter_count = 0
	while i < maxiter
		# calculate new constraint function values and Jacobian
	    jac!(J, cval, xnew)

		# check if tolerance met
		if norm(cval, Inf) < tol
			break
		end

	    # calculate right-hand side g = (J' c + μ*(xnew - xtilde))
	    g .= xnew .- xtilde

	    prev_dist2 = dot(g, g)	# for use below

	    gemv!('T', 1.0, J, cval, μ, g)
	    fill!(dx, 0.0)
	    r .= g

	    # do Newton step = (J'J + μI) \ r
	    pcg_flag, pcg_i = pcg!(μ, J, U, S, Vt, rank, dx, r, p, z, tmp_m, tol, maxiter_pcg)
	    pcg_iter_count += pcg_i
	    if pcg_flag > 0
	    	# do no further iterations if pcg exceeds the maximum iteration count
	    	flag = 2
	    	break
	    end

	    # do Armijo backtracking linesearch
	    p .= xnew
	    ar_dot = -dot(g, dx)			# step*gradient

	    prev_obj_val = dot(cval, cval) + μ*prev_dist2
	    α = 1.0

	    xnew .-= α.*dx
	    g .= xnew .- xtilde
	    dist2 = dot(g, g)
	    c!(cval, xnew)

	    armijo_count = 0
	    while dot(cval, cval) + μ*dist2 > prev_obj_val + 1e-4*α*ar_dot
	    	# update values with new step size
	    	α /= 2
	    	xnew .= p .- α.*dx
		    g .= xnew .- xtilde
		    dist2 = dot(g, g)
		    c!(cval, xnew)

		    # break with error if Armijo iteration count is excessive
	    	armijo_count += 1

	    	if armijo_count == 100
	    		flag = 3
	    		break
	    	end
	    end


	    # update iteration count and μ
		i += 1
		μ = min(μ*0.1, norm(cval))
		# μ *= 0.1
	end

	if i == maxiter
		flag = 1
	end

	return flag, i, pcg_iter_count

end