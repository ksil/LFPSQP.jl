@testset verbose=true "Projected CG                " begin
	n = 1000
	m = 10
	
	# construct some random data
	A = 0.01*randn(n, n)
	A = A*A' + 0.5*I

	b = randn(n)
	c = randn(m)

	F = qr(randn(n, m))
	U = Matrix(F.Q)

	x = zeros(n)
	λ = zeros(m)

	bigMat = [A U; U' zeros(m,m)]
	rhs = vcat(b, c)

	@testset "Accuracy" begin
		# do pcg with different tolerances and test for accuracy
		for tol in 10.0.^(-6:-1:-20)
			_, nr = LFPSQP.projcg!(x, λ, A, U, b, c, tol=tol)

			@test nr < tol
			@test norm(U'*x - c) < 1e-14
			@test norm(bigMat*vcat(x, λ) - rhs) < max(tol, 1e-13)
		end
	end

	@testset "Memory allocation" begin
		work = LFPSQP.ProjCGWork(n, m)
		mem = @allocated LFPSQP.projcg!(x, λ, A, U, b, c, tol=1e-20, work=work)

		@test mem == 0
	end

	@testset "Negative direction" begin
		# generate symmetric matrix with positive and negative eigenvalues
		F = qr(randn(n, n))
		S = Matrix(F.Q)
		Λ = vcat(rand(n-2*m) .+ 1, -1 .- rand(2*m))

		A = S*diagm(0 => Λ)*S'

		c = zeros(m)	# "negative direction" really only makes sense when c is 0

		i, nr = LFPSQP.projcg!(x, λ, A, U, b, c, tol=1e-20)

		@test isinf(nr)
		@test all(isnan.(λ))
		@test norm(U'*x - c) < 1e-14
		@test dot(x, A*x) <= 0.0
	end
end