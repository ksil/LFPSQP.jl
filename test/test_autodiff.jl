@testset verbose=true "Automatic differentiation   " begin
	n = 50
	m = 5

	# create example functions
	coeffs3 = randn(n, n, n)
	coeffs1 = randn(n)
	coeffs0 = randn();
	f(x) = sum(coeffs3[i,j,k]*x[i]*x[j]*x[k] for i in 1:n, j in 1:n, k in 1:n) + 
		dot(coeffs1, x) + coeffs0

	coeffs_c = randn(n-m+1, m)
	function c!(cval, x)
		for i in 1:m
			cval[i] = sum(exp.(-1.0.*view(coeffs_c, :, i).*view(x, i:n-m+i))) + x[i]^3*x[i+1]^2
		end
	end

	# ------------ test first-order derivatives -------------
	g = zeros(n)
	cval = zeros(m)
	cval2 = zeros(m)
	J = zeros(m, n)
	x0 = zeros(n)

	x = randn(n)
	xcopy = copy(x)		# make a copy to test for mutation

	actualg = copy(coeffs1)
	for l in 1:n
		actualg[l] += sum(coeffs3[l,j,k]*x[j]*x[k] for j in 1:n, k in 1:n) + 
			sum(coeffs3[i,l,k]*x[i]*x[k] for i in 1:n, k in 1:n) + 
			sum(coeffs3[i,j,l]*x[i]*x[j] for i in 1:n, j in 1:n)
	end

	actualJ = zeros(m, n)
	for i in 1:m
		actualJ[i, i:n-m+i] .= -view(coeffs_c, :, i) .* exp.(-1.0.*view(coeffs_c, :, i).*view(x, i:n-m+i))
		actualJ[i, i] += 3*x[i]^2*x[i+1]^2
		actualJ[i, i+1] += 2*x[i]^3*x[i+1]
	end

	# objective function gradient
	@testset "∇f" begin
		grad! = LFPSQP.generate_gradient(f, x0)
		grad!(g, x)

		@test all(g .≈ actualg)
		@test all(x .== xcopy)

		fill!(g, 0.0)
	end

	# objective function gradient with tape
	@testset "∇f (tape)" begin
		grad! = LFPSQP.generate_gradient_tape(f, x0)
		grad!(g, x)

		@test all(g .≈ actualg)
		@test all(x .== xcopy)

		fill!(g, 0.0)
	end

	# constraint Jacobian
	@testset "J_c" begin
		jac! = LFPSQP.generate_jacobian(c!, x0, m)
		jac!(J, cval, x)
		c!(cval2, x)

		@test all(J .≈ actualJ)
		@test all(cval .≈ cval2)
		@test all(x .== xcopy)

		fill!(J, 0.0)
		fill!(cval, 0.0)
	end

	# constraint Jacobian with tape
	@testset "J_c (tape)" begin
		jac! = LFPSQP.generate_jacobian_tape(c!, x0, m)
		jac!(J, cval, x)
		c!(cval2, x)

		@test all(J .≈ actualJ)
		@test all(cval .≈ cval2)
		@test all(x .== xcopy)

		fill!(J, 0.0)
		fill!(cval, 0.0)
	end


	# test Hessian vector product
	dest = zeros(n)
	dest2 = zeros(n)
	x0dual = zeros(Dual{nothing,Float64,1}, n)
	x0dual .= x0

	function actual_prod!(dest, src, x, λ)
		for l in 1:n
			dest[l] = sum(coeffs3[l,p,k]*x[k]*src[p] for k in 1:n, p in 1:n) + 
				sum(coeffs3[p,l,k]*x[k]*src[p] for k in 1:n, p in 1:n) + 
				sum(coeffs3[p,j,l]*x[j]*src[p] for j in 1:n, p in 1:n) + 
				sum(coeffs3[l,j,p]*x[j]*src[p] for j in 1:n, p in 1:n) + 
				sum(coeffs3[i,l,p]*x[i]*src[p] for i in 1:n, p in 1:n) + 
				sum(coeffs3[i,p,l]*x[i]*src[p] for i in 1:n, p in 1:n)
		end

		for i in 1:m
			dest[i:n-m+i] .+= λ[i].*view(coeffs_c, :, i).^2 .* exp.(-1.0.*view(coeffs_c, :, i).*view(x, i:n-m+i)) .* view(src, i:n-m+i)

			dest[i] += λ[i]*(6*x[i]*x[i+1]^2*src[i] + 6*x[i]^2*x[i+1]*src[i+1])
			dest[i+1] += λ[i]*(6*x[i]^2*x[i+1]*src[i] + 2*x[i]^3*src[i+1])
		end
	end

	grad_dual! = LFPSQP.generate_gradient(f, x0dual)
	jac_dual! = LFPSQP.generate_jacobian(c!, x0dual, m)

	hess_lag_vec! = LFPSQP.generate_hess_lag_vec(grad_dual!, jac_dual!, x0, m::Int)

	# try a few different products
	@testset "Hessian-vector product" begin
		for trial = 1:5
			λ = randn(m)
			λcopy = copy(λ)
			v = randn(n)
			vcopy = copy(v)

			hess_lag_vec!(dest, v, x, λ)
			actual_prod!(dest2, v, x, λ)

			@test all(dest .≈ dest2)
			@test all(λ .== λcopy)
			@test all(v .== vcopy)
		end
	end

end