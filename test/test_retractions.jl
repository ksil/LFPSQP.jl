@testset verbose=true "Retractions                 " begin
	@testset "qrupdate" begin
		n = 3
		A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.3]
		u = [1.0; 2.0; 3.0]
		v = [1.0; 2.0; 4.0]
		w = zeros(n)

		B = A + u*v'

		F = qr(A)
		Q = Matrix(F.Q)
		R = Matrix(F.R)

		@show R
		@show w

		LFPSQP.qrupdate!(Q, R, u, v, w)

		@show R
		@show w

		@show A
		@show B
		@show Q*R
		@show Q*Q'

		@test all(B .≈ Q*R)
		@test all(abs.(Q*Q' .- Matrix(I, n, n)) .< 1e-15)
	end


	# generate random spheres such that x0 is a solution
	n = 1000
	m = 70

	x0 = randn(n)
	val2 = zeros(m)

	Rs = rand(m) .+ 1
	centers = zeros(n, m)

	for i = 1:m
	    dir = randn(n)
	    normalize!(dir)
	    centers[:,i] = x0 + Rs[i]*dir
	end

	sphere(x, c, R) = dot(x .- c, x .- c) - R^2

	function c!(val, x)
	    for i = 1:m
	        val[i] = sphere(x, centers[:,i], Rs[i])
	    end
	end

	function jac!(J, x)
	    for i = 1:m
	    	J[i,:] .= 2.0.*(x .- centers[:,i])
	    end
	end

	# do SVD
	cval = zeros(m)
	cval2 = zeros(m)
	J = zeros(m,n)
	jac!(J, x0)

	U, S, Vt = svd(Array(J'))
	Vt = Array(Vt);

	nrwork =LFPSQP.NRWork(m)

	# generate random step and project onto tangent plane
	step = randn(n)
	step .= 0.1 .* step ./ norm(step)
	step -= U*(U'*step)

	xtilde = x0 + step
	xtilde_copy = copy(xtilde)
	xnew = zeros(n)

	@testset "Newton retraction" begin
		for tol in [1e-4, 1e-6, 1e-8]
			flag, i = LFPSQP.NR!(cval, xnew, c!, xtilde, U, S, Vt, tol, 10000, nrwork)
			c!(cval2, xnew)

			@show i
			@test norm(cval, Inf) < tol
			@test all(cval .== cval2)			# cval should be c! evaluated at xnew
			@test all(xtilde .== xtilde_copy)
			@test abs(dot(step, xnew - xtilde)) < 1e-6
		end
	end
end