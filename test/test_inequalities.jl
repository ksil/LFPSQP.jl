@testset verbose=true "Inequalities                " begin

	n = 16
	m = 5

	# no bounds, lower bound, upper bound, both bounds
	divsize = n ÷ 4
	xl = [-Inf.*ones(divsize); randn(divsize); -Inf.*ones(divsize); randn(divsize)]
	xu = [Inf.*ones(divsize); Inf.*ones(divsize); randn(divsize); randn(divsize)]

	x = [randn(n÷4);
		xl[divsize+1:2*divsize] .+ rand(0:2, divsize);
		xu[2*divsize+1:3*divsize] .- rand(0:2, divsize);
		xl[3*divsize+1:4*divsize] .+ rand(divsize).*(xu[3*divsize+1:4*divsize] .- xl[3*divsize+1:4*divsize])]

	xaug = zeros(2*n)
	xaug[1:n] .= x

	idata = LFPSQP.InequalityData(xl, xu)

	# test aspects of Inequality data
	@testset "InequalityData" begin
		@test all(idata.q .≈ [zeros(3*divsize); ones(divsize)])
		@test all(idata.r .≈ [zeros(divsize);
								xl[divsize+1:2*divsize];
								xu[2*divsize+1:3*divsize];
								xl[3*divsize+1:4*divsize]./2 .+ xu[3*divsize+1:4*divsize]./2])
		@test all(idata.s .≈ [zeros(divsize); -ones(divsize); ones(2*divsize)])
		@test all(idata.t .≈ [zeros(divsize);
								xl[divsize+1:2*divsize];
								xu[2*divsize+1:3*divsize];
								(xu[3*divsize+1:4*divsize] .- xl[3*divsize+1:4*divsize]).^2 ./ 4])

		@test all(idata.isline .== vcat(trues(divsize), falses(3*divsize)))
		@test all(idata.isparabola .== vcat(falses(divsize), trues(2*divsize), falses(divsize)))
	end

	# test y generation satisfies inequalities
	@testset "Initial y" begin
		LFPSQP.generate_initial_y!(xaug, idata)

		xview = view(xaug, 1:n)
		yview = view(xaug, n+1:2*n)

		@test all(isapprox.(idata.q.*(xview .- idata.r).^2 .+ (1.0 .- idata.q.^2).*xview .+
			idata.s.*(yview .- idata.r).^2 .- (1.0 .- idata.s.^2).*yview .- idata.t, 0.0, atol=1e-15))

		cvalaug = zeros(n+m)
		LFPSQP.calculate_h!(cvalaug, xaug, idata)

		@test all(isapprox.(cvalaug, 0.0, atol=1e-15))
	end

	# generate random decomposition
	Jct = randn(n, m)

	∇hx = 2.0*idata.q.*(x .- idata.r) .+ (1.0 .- idata.q.^2)
	∇hy = 2.0*idata.s.*(xaug[n+1:end] .- idata.r) .- (1.0 .- idata.s.^2)

	S = sqrt.(∇hx.^2 .+ ∇hy.^2)
	Dx = ∇hx ./ S
	Dy = ∇hy ./ S

	R = Dx .* Jct

	PJct = zeros(2*n, m)
	PJct[1:n, :] .= (1.0 .- Dx.*Dx) .* Jct
	PJct[n+1:2*n, :] .= -1.0 .* Dy.*Dx .* Jct

	U, Σ, V = svd(PJct)
	Vt = Matrix(V')

	bigA = [diagm(∇hx) Jct; diagm(∇hy) zeros(n, m)]
	bigQ = hcat([diagm(Dx); diagm(Dy)], U)
	bigR = [diagm(S) R; zeros(m, n) diagm(Σ)*Vt]

	idecomp = LFPSQP.InequalityDecomp(U, Σ, Vt, [Array{Float64}(undef, n) for i in 1:3]..., Jct, m)

	# test InequalityDecomposition
	@testset "InequalityDecomposition" begin

		LFPSQP.inequality_gradient!(idecomp, xaug, idata)

		@test all(isapprox.(idecomp.Dx, Dx, atol=1e-15))
		@test all(isapprox.(idecomp.Dy, Dy, atol=1e-15))
		@test all(isapprox.(idecomp.S, S, atol=1e-15))

		@test all(isapprox.(bigA, bigQ*bigR, atol=1e-14))					# decomposition is correct
		@test all(isapprox.(bigQ'*bigQ, Matrix(I, n+m, n+m), atol=1e-14))	# orthogonality
	end

	@testset "Multiplication" begin
		v = randn(n+m)
		w = randn(2*n)

		destv = zeros(2*n)
		destw = zeros(n+m)

		# projections
		mul!(destv, LFPSQP.InequalityDecompProject(idecomp), v)
		@test all(isapprox.(destv, bigQ*v, atol=1e-15))

		mul!(destw, LFPSQP.InequalityDecompProject(idecomp)', w)
		@test all(isapprox.(destw, bigQ'*w, atol=1e-15))

		destv .= 1.0
		mul!(destv, LFPSQP.InequalityDecompProject(idecomp), v, 2.0, 3.0)
		@test all(isapprox.(destv, 2*bigQ*v .+ 3, atol=1e-15))


		# Jacobians
		mul!(destv, idecomp, v)
		@test all(isapprox.(destv, bigA*v, atol=1e-15))

		mul!(destw, idecomp', w)
		@test all(isapprox.(destw, bigA'*w, atol=1e-15))

		destv .= 1.0
		mul!(destv, idecomp, v, 2.0, 3.0)
		@test all(isapprox.(destv, 2*bigA*v .+ 3, atol=3e-15))


		# test lower rank
		idecomp.rank = m - 2

		v = randn(n+m-2)
		destv = zeros(2*n)
		destw = zeros(n+m-2)

		mul!(destv, LFPSQP.InequalityDecompProject(idecomp), v)
		@test all(isapprox.(destv, bigQ[:, 1:n+m-2]*v, atol=1e-15))

		mul!(destw, LFPSQP.InequalityDecompProject(idecomp)', w)
		@test all(isapprox.(destw, bigQ[:, 1:n+m-2]'*w, atol=1e-15))

		destv .= 1.0
		mul!(destv, LFPSQP.InequalityDecompProject(idecomp), v, 2.0, 3.0)
		@test all(isapprox.(destv, 2*bigQ[:, 1:n+m-2]*v .+ 3, atol=1e-15))

		idecomp.rank = m
	end

	@testset "Lagrange multipliers" begin
		d = randn(2*n)
		Qt∇f = zeros(n+m)
		mul!(Qt∇f, LFPSQP.InequalityDecompProject(idecomp)', d)

		λ = zeros(n+m)
		λ_kkt = view(λ, n+1:n+m)
		λy_kkt = view(λ, 1:n)

		LFPSQP.calculate_λ_kkt!(λ_kkt, λy_kkt, Qt∇f, idecomp)

		@test all(isapprox.(λ, bigA \ d, atol=1e-14))
	end

	@testset "Hessian action" begin

		A = randn(n, n)
		C = randn(n, n, m)
		λ_kkt = randn(m)
		λy_kkt = randn(n)

		dest = zeros(2*n)
		v = randn(2*n)

		function hess_lag_vec!(dest, src, x, λ)
			dest .= A*src + sum(λ[i]*C[:,:,i]*src for i in 1:m)
		end

		LFPSQP.augmented_hess_lag_vec!(dest, v, hess_lag_vec!, xaug, λ_kkt, λy_kkt, idata)

		H = A + sum(λ_kkt[i]*C[:,:,i] for i in 1:m)
		bigH = [H+2*diagm(λy_kkt.*idata.q) zeros(n, n); zeros(n,n) 2*diagm(λy_kkt.*idata.s)]

		@test all(isapprox.(dest, bigH*v, atol=1e-14))
	end


	@testset "y retraction" begin
		# take step in tangent plane
		d = randn(2*n)
		tmp_m = zeros(n+m)
		ineqproject = LFPSQP.InequalityDecompProject(idecomp)

		mul!(tmp_m, ineqproject', d)				# tmp_m = Q'd
		mul!(d, ineqproject, tmp_m, -1.0, 1.0)		# d = d - Q*tmp_m

		# test really long step length
		xnewaug = xaug .+ d
		xaugcopy = copy(xaug)
		LFPSQP.y_retract!(xnewaug, xaug, idata)

		# ensure y retraction worked and that inequality constraints are 0
		cvalaug = zeros(n+m)
		LFPSQP.calculate_h!(cvalaug, xnewaug, idata)

		@test all(isapprox.(cvalaug, 0.0, atol=1e-13))
		@test all(xaug .== xaugcopy)					# ensure xaug not overwritten
	end
end