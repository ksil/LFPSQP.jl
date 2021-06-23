@testset verbose=true "Linesearch                  " begin

	f = x -> x[1]^2

	x = [-0.23]
	xnew = copy(x)
	d = [1.0]
	g = 2*x
	fval = f(x)
	cval = zeros(0)

	@testset "Armijo" begin
		flag, tot_iter1, tot_iter2, newf, f_diff, step_diff, α = LFPSQP.armijo!(xnew, x, d, g, f, fval, LFPSQP.Euclidean(), cval, c!, LFPSQPParams(linesearch=LFPSQP.armijo), LFPSQP.ArmijoWork(1))

		@test flag == tot_iter1 == tot_iter2 == 0
		@test x[1] == -0.23							# ensure value not changed
		@test newf ≈ f(xnew)
		@test f_diff ≈ fval - newf
		@test step_diff ≈ α ≈ 0.25
	end

	@testset "Exact" begin
		flag, tot_iter1, tot_iter2, newf, f_diff, step_diff, α = LFPSQP.exact_linesearch!(xnew, x, d, f, fval, LFPSQP.Euclidean(), cval, c!, LFPSQPParams(linesearch=LFPSQP.armijo), LFPSQP.ExactLinesearchWork(1))
		
		@test flag == tot_iter1 == tot_iter2 == 0
		@test x[1] == -0.23							# ensure value not changed
		@test newf ≈ f(xnew)
		@test f_diff ≈ fval - newf
		@test step_diff ≈ α && isapprox(α, 0.23, atol=1e-6)
	end
end