struct NRWork
	D::Array{Float64, 2}
	tmp_m::Vector{Float64}
	tmp_m2::Vector{Float64}
	dc::Vector{Float64}
end

NRWork(m::Int) = NRWork(Array{Float64}(undef, m, m), [Array{Float64}(undef, m) for i in 1:3]...)


function NR!(cval, xnew, c!, xtilde, U, S, Vt, tol, maxiter::Int, work::NRWork=NRWork(length(S)))
	#= performs Newton-Raphson retraction for c(xtilde + U d) using the Jacobian V Σ'
	which is evaluated at x

	termination criterion is tolerance in inf norm of c!

	INPUT
	cval - contraint value vector
	xnew - vector in which to store the result
	c! - constraint function of the form c!(cval, x) where y is overwritten
	xtilde - current value of x at step in tangent space
	U, S, Vt - SVD decomposition
	tol - function tolerance
	maxiter - maximum number of allowable iterations
	work - NRWork struct for storage

	OUTPUT
	flag - 0 = success, 1 = maxiter reached
	i - total number of iterations taken

	OVERWRITTEN
	cval, xnew

	=#

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