function generate_sphere_system(n, m)
	x0 = zeros(n)

	Rs = rand(m) .+ 1
	centers = zeros(n, m)

	for i = 1:m
	    dir = randn(n)
	    normalize!(dir)
	    centers[:,i] = x0 + Rs[i]*dir
	end

	sphere(x, c, R) = dot(x, x) + dot(c, c) - 2*dot(x, c) - R^2

	function c!(cval, x)
	    for i = 1:m
	        cval[i] = sphere(x, view(centers, :, i), Rs[i])
	    end
	end

	function jac!(J, cval, x)
		fill!(J, 0.0)

	    for i = 1:m
	    	cval[i] = sphere(x, view(centers, :, i), Rs[i])
	    	J[i,:] .= 2.0.*(x .- view(centers, :, i))
	    end
	end

	return x0, c!, jac!
end


function generate_sin_system(n, m)
	x0 = zeros(n)

	function c!(cval, x)
	    for i = 1:m
	        cval[i] = x[2*i] - sin(x[2*i-1])
	    end
	end

	function jac!(J, cval, x)
		fill!(J, 0.0)

	    for i = 1:m
	    	cval[i] = x[2*i] - sin(x[2*i-1])
	    	J[i, 2*i] = 1.0
	    	J[i, 2*i-1] = -cos(x[2*i-1])
	    end
	end

	return x0, c!, jac!
end



@testset verbose=true "Retractions                 " begin
	
	n = 1000
	m = 100

	x0, c!, jac! = generate_sin_system(n, m)

	# do SVD
	cval = zeros(m)
	cval2 = zeros(m)
	J = zeros(m,n)
	jac!(J, cval, x0)

	U, S, Vt = svd(Array(J'))
	Vt = Array(Vt);

	nr = LFPSQP.NR(U, S, Vt, 1.0, 1000, LFPSQP.NRWork(m), false, LFPSQP.InequalityData())
	pp = LFPSQP.ProjPenalty(jac!, U, S, Vt, m, 0.01, 1.0, 100, 200, LFPSQP.ProjPenaltyWork(m, n, m, n), false, LFPSQP.InequalityDecomp(), LFPSQP.InequalityData())

	# generate random step and project onto tangent plane
	step = randn(n)
	step -= U*(U'*step)
	step .*= 5.0 / norm(step)

	xtilde = x0 + step
	xtilde_copy = copy(xtilde)
	xnew = zeros(n)

	c!(cval, xtilde)
	# @show norm(cval)


	@testset "Newton retraction" begin
		for tol in [1e-6, 1e-8]
			nr.tol = tol

			flag, i, _ = LFPSQP.retract!(cval, xnew, c!, xtilde, x0, nr)
			c!(cval2, xnew)

			# @show flag, i
			@test flag == 0 && norm(cval, Inf) < tol
			@test all(cval .== cval2)			# cval should be c! evaluated at xnew
			@test all(xtilde .== xtilde_copy)
			@test abs(dot(step, xnew - xtilde)) < 1e-6
		end
	end

	@testset "CG iteration" begin
		# solve (μI + J'J)x = b
		J = randn(m, n)
		p = zeros(n)
		z = zeros(n)
		tmp_m = zeros(m)
		tol = 1e-6
		maxiter = 100

		for μ in [1e-1, 1e-2, 1e-4]
			x = zeros(n) # initial guess
			b = randn(n)
			r = copy(b)

			M! = LFPSQP.no_precondition

			flag, i = LFPSQP.pcg!(μ, J, M!, x, r, p, z, tmp_m, tol, maxiter)

			@test flag == 0
			@test norm(r) < tol
			@test norm(μ.*x + J'*(J*x) - b) < tol

			# test exact preconditioning
			x = zeros(n) # initial guess
			b = randn(n)
			r = copy(b)

			exact_inverse = inv(μ*Matrix(I, n, n) + J'*J)
			M!(z, r) = z .= (μ*Matrix(I, n, n) + J'*J) \ r

			flag, i = LFPSQP.pcg!(μ, J, M!, x, r, p, z, tmp_m, tol, maxiter)

			@test flag == 0 && i == 1
			@test norm(r) < tol
			@test norm(μ.*x + J'*(J*x) - b) < tol
		end
	end


	@testset "Project penalty retraction" begin
		for tol in [1e-6, 1e-8, 1e-10]
			pp.tol = tol

			flag, i, pcg_i = LFPSQP.retract!(cval, xnew, c!, xtilde, x0, pp)
			c!(cval2, xnew)

			# @show flag, i, pcg_i
			@test flag == 0 && norm(cval, Inf) < tol
			@test all(cval .== cval2)			# cval should be c! evaluated at xnew
			@test all(xtilde .== xtilde_copy)
			@test norm(step) >= norm(xnew - x0) - tol
		end
	end
end