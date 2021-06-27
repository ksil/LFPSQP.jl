struct NRWork
	D::Array{Float64, 2}
	tmp_m::Vector{Float64}
	tmp_m2::Vector{Float64}
	dc::Vector{Float64}
end

NRWork(m::Int) = NRWork(Array{Float64}(undef, m, m), [Array{Float64}(undef, m) for i in 1:3]...)

mutable struct NR
	U::Array{Float64, 2}
	Σ::Array{Float64, 1}
	Vt::Array{Float64, 2}
	tol::Float64
	maxiter::Int
	work::NRWork
	ineq::Bool
	idata::InequalityData
end

struct ProjPenaltyWork
	J::Array{Float64, 2}
	tmp_m::Vector{Float64}
	r::Vector{Float64}
	p::Vector{Float64}
	z::Vector{Float64}
	dx::Vector{Float64}
	g::Vector{Float64}
	cvalaug::Vector{Float64}
end

ProjPenaltyWork(m::Int, n::Int, m_ineq::Int, n_ineq::Int) = ProjPenaltyWork(Array{Float64}(undef, m, n), 
	Array{Float64}(undef, m_ineq), [Array{Float64}(undef, n_ineq) for i in 1:5]..., Array{Float64}(undef, m_ineq))

mutable struct ProjPenalty{F}
	jac!::F
	U::Array{Float64, 2}
	Σ::Array{Float64, 1}
	Vt::Array{Float64, 2}
	rank::Int
	μ0::Float64
	tol::Float64
	maxiter::Int
	maxiter_pcg::Int
	work::ProjPenaltyWork
	ineq::Bool
	idecomp::InequalityDecomp
	idata::InequalityData
end

struct Euclidean
end

struct YRetract
	idata::InequalityData
end


# -------------------------- Retraction functions -------------------------------------

function retract!(cval, xnew, c!, xtilde, x, method::Euclidean)
	xnew .= xtilde

	return 0, 0, 0
end

function retract!(cval, xnew, c!, xtilde, x, method::YRetract)
	xnew .= xtilde
	y_retract!(xnew, x, method.idata)

	return 0, 0, 0
end


function retract!(cval, xnew, c!, xtilde, x, method::NR)
	#= performs Newton-Raphson retraction for c(xtilde + U d) using the Jacobian V Σ'
	which is evaluated at x

	termination criterion is tolerance in inf norm of c!

	INPUT
	cval - contraint value vector
	xnew - vector in which to store the result
	c! - constraint function of the form c!(cval, x) where y is overwritten
	xtilde - current value of x at step in tangent space
	x - previous iterate
	U, Σ, Vt - SVD decomposition
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
	Σ = method.Σ
	Vt = method.Vt
	tol = method.tol
	maxiter = method.maxiter
	work = method.work

	# extract quantities from NRWork
	D = work.D    			# m x m array to store inverse of Jacobian of c(xtilde + Ud)
	tmp_m = work.tmp_m
	tmp_m2 = work.tmp_m2
	dc = work.dc

	m = length(Σ)
	xnew .= xtilde

	if method.ineq
		y_retract!(xnew, x, method.idata)
		c!(cval, view(xnew, 1:length(xnew)÷2))
	else
		c!(cval, xnew)
	end

	# calculate inverse Jacobian Σ^(-1) V'
	for j in 1:m
		@fastmath @inbounds @simd for k in 1:m
			D[k,j] = Vt[k,j] / Σ[k]
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
		if method.ineq
			y_retract!(xnew, x, method.idata)
			c!(tmp_m2, view(xnew, 1:length(xnew)÷2))
		else
			c!(tmp_m2, xnew)
		end

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

function pcg!(μ, J, M!, x, r, p, z, tmp_m, tol, maxiter)
	#= performs preconditioned conjugate gradient to solve Ax = b with
		A = J' J + μI
		preconditioned with M!

		INPUT
		μ - factor for projection
		J - the Jacobian of the constraints
		M! - function for preconditioning of the form M!(z, r)
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

	norm_res = Inf
	ρ = 1.0               	# set ρ and p initialy to 1.0 and 0.0, as done in IterativeSolvers.jl
	fill!(p, 0.0)

	i = 0
	while norm_res > tol && i < maxiter
		# precondition - z = "A^{-1}" r
		M!(z, r)

		# update ρ = r^T z
		ρ_prev = ρ
		ρ = dot(z, r)

		# update β and direction p
		β = ρ / ρ_prev
		p .= z .+ β.*p

		# store A*p in z
		z .= p
		mul!(tmp_m, J, p)
		mul!(z, J', tmp_m, 1.0, μ)
		# gemv!('N', 1.0, J, p, 0.0, tmp_m)
		# gemv!('T', 1.0, J, tmp_m, μ, z)

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

function proj_precondition!(z, r, μ, U, Σ, rank, tmp_m)
	z .= r
	kgemv!('T', rank, 1.0, U, r, 0.0, tmp_m)
	@fastmath @inbounds @simd for j in 1:rank
		tmp_m[j] *= Σ[j]*Σ[j] / (μ + Σ[j]*Σ[j])
	end
	kgemv!('N', rank, -1/μ, U, tmp_m, 1/μ, z)

	return z
end

function no_precondition(z, r)
	z .= r

	return z
end

function retract!(cval, xnew, c!, xtilde, x, method::ProjPenalty{F}) where F

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
	x - previous iterate
	jac! - function to calculate Jacobian of the form jac!(J, cval, x)
	U, Σ, Vt - SVD decomposition of Jacobian at xtilde
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
	Σ = method.Σ
	Vt = method.Vt
	rank = method.rank
	μ0 = method.μ0
	tol = method.tol
	maxiter = method.maxiter
	maxiter_pcg = method.maxiter_pcg
	work = method.work
	idecomp = method.idecomp
	idata = method.idata

	# unpack work variables
	J = work.J
	tmp_m = work.tmp_m
	r = work.r
	p = work.p
	z = work.z
	dx = work.dx
	g = work.g
	cvalaug = work.cvalaug

	# matrix to use for multiplication
	fulljac = method.ineq ? idecomp' : J

	flag = 0 # return flag

	# initialize x vectors and calculate constraint functions
	xnew .= xtilde
	μ = μ0

	# store length of "x" part
	n = method.ineq ? length(xnew) ÷ 2 : length(xnew)
	m = length(Σ)

	i = 0
	pcg_iter_count = 0
	while i < maxiter
		# calculate new constraint function values and Jacobian
		jac!(J, cval, view(xnew, 1:n))
    	curtol = norm(cval, Inf)

	    if method.ineq
	    	inequality_gradient!(idecomp, xnew, idata)

	    	# update c Jacobian in idecomp
	    	transpose!(idecomp.Jct, J)

	    	# calculate inequality constraint norm
	    	calculate_h!(cvalaug, xnew, idata)

	    	curtol = max(curtol, norm(cvalaug, Inf))
	    end

	    # fill last m values of cvalaug with cval
	    cvalaug[end-m+1:end] .= cval

		# check if tolerance met
		if curtol < tol
			break
		end

	    # calculate right-hand side g = (J' c + μ*(xnew - xtilde))
	    g .= xnew .- xtilde

	    prev_obj_val = dot(cvalaug, cvalaug) + μ*dot(g, g)	# for use below

	    # gemv!('T', 1.0, fulljac, cvalaug, μ, g)
	    mul!(g, fulljac', cvalaug, 1.0, μ)
	    fill!(dx, 0.0)
	    r .= g

	    # do Newton step = (J'J + μI) \ r
	    # M! = (z, r) -> proj_precondition!(z, r, μ, U, Σ, rank, tmp_m)
	    pcg_flag, pcg_i = pcg!(μ, fulljac, no_precondition, dx, r, p, z, tmp_m, tol, maxiter_pcg)
	    pcg_iter_count += pcg_i
	    if pcg_flag > 0
	    	# do no further iterations if pcg exceeds the maximum iteration count
	    	flag = 2
	    	break
	    end

	    # do Armijo backtracking linesearch
	    p .= xnew
	    ar_dot = -dot(g, dx)			# step*gradient

	    α = 1.0

	    xnew .-= α.*dx
	    g .= xnew .- xtilde
	    dist2 = dot(g, g)
	    c!(cval, view(xnew, 1:n))

	    # update inequality constraint function values
	    if method.ineq
	    	calculate_h!(cvalaug, xnew, idata)
	    end

	    cvalaug[end-m+1:end] .= cval


	    armijo_count = 0
	    while dot(cvalaug, cvalaug) + μ*dist2 > prev_obj_val + 1e-4*α*ar_dot
	    	# update values with new step size
	    	α /= 2
	    	xnew .= p .- α.*dx
		    g .= xnew .- xtilde
		    dist2 = dot(g, g)

		    c!(cvalaug, view(xnew, 1:n))

		    # update inequality constraint function values
		    if method.ineq
		    	calculate_h!(cvalaug, xnew, idata)
		    end

	    	cvalaug[end-m+1:end] .= cval

		    # break with error if Armijo iteration count is excessive
	    	armijo_count += 1

	    	if armijo_count == 100
	    		flag = 3
	    		break
	    	end
	    end


	    # update iteration count and μ
		i += 1
		μ = min(μ*0.1, norm(cvalaug))
		# μ *= 0.1
	end

	if i == maxiter
		flag = 1
	end

	return flag, i, pcg_iter_count

end



# ---------------------------- Inequality retractions ---------------------------------------

# retract onto the inequality constraints (y) first and overwrite results in xnewaug
# xnewaug - "xtilde" to be overwritten
# xaug - point at which retraction is based
# idata
function y_retract!(xnewaug, xaug, idata::InequalityData)
	n = length(xaug) ÷ 2

	xnew = view(xnewaug, 1:n)
	ynew = view(xnewaug, n+1:2*n)
	x = view(xaug, 1:n)
	y = view(xaug, n+1:2*n)

	for i in 1:n
		if idata.isline[i]
			# don't need to do anything since step should be in tangent plane
			# but can update to avoid numerical drift
			xnew[i] = ynew[i]
		elseif idata.isparabola[i]
			# retract back to parabola with a second-order retractor
			s = idata.s[i]
			r = idata.r[i]

			g = (-s, -2(y[i] - r))			# gradient at x
			ng = norm(g)
			ux = x[i] - xnew[i] + g[1]/ng
			uy = y[i] - ynew[i] + g[2]/ng

			# solve quadratic and calculate step length, γ, back to parabola
			a = s*uy^2
			b = ux + 2*s*(ynew[i] - r)*uy
			c = xnew[i] + s*(ynew[i] - r)^2 - r

			a1 = -b/(2*a)
			a2 = sqrt(b^2 - 4*a*c)/(2*a)

			γ = min(a1 + a2, a1 - a2)

			# update xnew and ynew
			xnew[i] += γ*ux
			ynew[i] += γ*uy
		else # iscircle
			# retract to circle via simple projection
			c = idata.r[i]
			ρ = sqrt(idata.t[i])

			dist = sqrt((xnew[i] - c)^2 + (ynew[i] - c)^2)

			ynew[i] = c + ρ*(ynew[i] - c)/dist
			xnew[i] = c + ρ*(xnew[i] - c)/dist
		end
	end

	return xnew
end