struct InequalityData
	q::Vector{Float64}
	r::Vector{Float64}
	s::Vector{Float64}
	t::Vector{Float64}
	isline::BitVector
	isparabola::BitVector
end

mutable struct InequalityDecomp
	U::Array{Float64, 2}
	Σ::Vector{Float64}
	Vt::Array{Float64}
	Dx::Vector{Float64}
	Dy::Vector{Float64}
	S::Vector{Float64}
	Jct::Array{Float64, 2}
	rank::Int
end

struct InequalityDecompAdjoint
	idecomp::InequalityDecomp
end

struct InequalityDecompProject
	idecomp::InequalityDecomp
end

struct InequalityDecompProjectAdjoint
	idecomp::InequalityDecomp
end

Base.adjoint(ip::InequalityDecompProject) = InequalityDecompProjectAdjoint(ip.idecomp)
Base.adjoint(idecomp::InequalityDecomp) = InequalityDecompAdjoint(idecomp)
Base.adjoint(ipadj::InequalityDecompProjectAdjoint) = InequalityDecompProject(ipadj.idecomp)
Base.adjoint(idecompadj::InequalityDecompAdjoint) = idecompadj.idecomp

# construct InequalityData struct using lower and upper bounds for x
function InequalityData(xl, xu)
	n = length(xl)

	if length(xu) != n
		error("xl and xu are of different lengths")
	end
	
	# allocate vectors
	q = Array{Float64}(undef, n)
	r = Array{Float64}(undef, n)
	s = Array{Float64}(undef, n)
	t = Array{Float64}(undef, n)
	isline = falses(n)
	isparabola = falses(n)

	@inbounds for i in 1:n
		linf = isinf(xl[i])
		uinf = isinf(xu[i])

		if linf && uinf
			q[i] = 0
			r[i] = 0
			s[i] = 0
			t[i] = 0
			isline[i] = true
		elseif !linf && uinf
			q[i] = 0
			r[i] = xl[i]
			s[i] = -1.0
			t[i] = xl[i]
			isparabola[i] = true
		elseif linf && !uinf
			q[i] = 0
			r[i] = xu[i]
			s[i] = 1.0
			t[i] = xu[i]
			isparabola[i] = true
		else
			q[i] = 1.0
			r[i] = (xu[i] + xl[i])/2
			s[i] = 1.0
			t[i] = (xu[i] - xl[i])^2 / 4
		end
	end

	return InequalityData(q, r, s, t, isline, isparabola)
end

# empty inequality data
InequalityData() = InequalityData(zeros(0), zeros(0))
InequalityDecomp() = InequalityDecomp(zeros(0,0), [zeros(0) for i in 1:5]..., zeros(0,0), 0)

# fills the second half of xaug (equal to y) with initial values that satisfy constraint manifold
function generate_initial_y!(xaug, idata::InequalityData)
	n = length(idata.q)

	x = view(xaug, 1:n)
	y = view(xaug, n+1:2*n)

	@inbounds for i in 1:n
		if idata.isline[i]
			y[i] = x[i]
		elseif idata.isparabola[i]
			y[i] = sqrt(max(-(x[i] - idata.t[i])/idata.s[i], 0.0)) + idata.r[i]
		else # is circle
			y[i] = sqrt(max(idata.t[i] - (x[i] - idata.r[i])^2, 0.0)) + idata.r[i]
		end
	end

	return xaug
end

# overwrites the first n values of cvalaug with the values of the bound constraints
function calculate_h!(cvalaug, x, idata::InequalityData)
	n = length(x) ÷ 2

	xview = view(x, 1:n)
	yview = view(x, n+1:2*n)

	cvalaug[1:n] .= idata.q.*(xview .- idata.r).^2 .+ (1.0 .- idata.q.^2).*xview .+
				idata.s.*(yview .- idata.r).^2 .- (1.0 .- idata.s.^2).*yview .- idata.t

	return cvalaug
end

# overwrites Dx, Dy, and S using inequality values
function inequality_gradient!(idecomp::InequalityDecomp, x, idata::InequalityData)
	n = length(x) ÷ 2

	# unpack
	Dx = idecomp.Dx
	Dy = idecomp.Dy
	S = idecomp.S

	# calculate gradients
	Dx .= 2.0.*idata.q.*(view(x, 1:n) .- idata.r) .+ (idata.q .== 0.0)
	Dy .= 2.0.*idata.s.*(view(x, n+1:2*n) .- idata.r) .- (idata.s .== 0.0)
	
	# normalize columns of Dx and Dy
	S .= sqrt.(Dx.*Dx .+ Dy.*Dy)
	Dx ./= S
	Dy ./= S
end

# function that defines map used in projected CG Newton steps
function augmented_hess_lag_vec!(dest, src, hess_lag_vec!, x, λ_kkt, λy_kkt, idata::InequalityData)
	n = length(x) ÷ 2

	src_x = view(src, 1:n)
	src_y = view(src, n+1:2*n)
	dest_x = view(dest, 1:n)
	dest_y = view(dest, n+1:2*n)

	# calculate Hessian action with respect to x
	hess_lag_vec!(dest_x, src_x, view(x, 1:n), λ_kkt)

	# calculate Hessian action with respect to y
	dest_x .+= 2 .* λy_kkt .* idata.q .* src_x
	dest_y .= 2 .* λy_kkt .* idata.s .* src_y
end

# define projection operators for InequalityDecompProject
function LinearAlgebra.mul!(dest, ip::InequalityDecompProject, v)
	idecomp = ip.idecomp
	n = length(idecomp.Dx)
	rank = idecomp.rank
	Dx = idecomp.Dx
	Dy = idecomp.Dy

	# dest = U[:, 1:rank]*v[n+1:n+rank]
	mul!(dest, view(idecomp.U, :, 1:rank), view(v, n+1:n+rank))

	# dest = dest + [Dx; Dy]*v[1:n]
	dest[1:n] .+= Dx .* view(v, 1:n)
	dest[n+1:2*n] .+= Dy .* view(v, 1:n)

	return dest
end

# dest = idecomp*v*a + dest*b
function LinearAlgebra.mul!(dest, ip::InequalityDecompProject, v, a, b)
	idecomp = ip.idecomp
	n = length(idecomp.Dx)
	rank = idecomp.rank
	Dx = idecomp.Dx
	Dy = idecomp.Dy

	# dest = a*U*v[n+1:n+rank] + b*dest
	mul!(dest, view(idecomp.U, :, 1:rank), view(v, n+1:n+rank), a, b)

	# dest = dest + a*[Dx; Dy]*v[1:n]
	dest[1:n] .+= a .* Dx .* view(v, 1:n)
	dest[n+1:2*n] .+= a .* Dy .* view(v, 1:n)

	return dest
end

# transpose projection
function LinearAlgebra.mul!(dest, ipa::InequalityDecompProjectAdjoint, v)
	idecomp = ipa.idecomp
	n = length(idecomp.Dx)
	rank = idecomp.rank
	Dx = idecomp.Dx
	Dy = idecomp.Dy

	# dest[1:n] = [Dx; Dy]*v[1:n]
	dest[1:n] .= Dx .* view(v, 1:n)
	dest[1:n] .+= Dy .* view(v, n+1:2*n)

	# dest[n+1:n+rank] = U'*v
	mul!(view(dest, n+1:n+rank), view(idecomp.U, :, 1:rank)', v)

	return dest
end

# define multiplication operators for InequalityDecomp
function LinearAlgebra.mul!(dest, idecomp::InequalityDecomp, v)
	n = length(idecomp.Dx)
	Jct = idecomp.Jct
	m = size(Jct, 2)
	Dx = idecomp.Dx
	Dy = idecomp.Dy
	S = idecomp.S

	# dest[1:n] = Jct*v[n+1:n+m]
	mul!(view(dest, 1:n), Jct, view(v, n+1:n+m))

	# dest += [Dx; Dy]*S * v[1:n]
	dest[1:n] .+= Dx .* S .* view(v, 1:n)
	dest[n+1:2*n] .= Dy .* S .* view(v, 1:n)

	return dest
end

# dest = idecomp*v*a + dest*b
function LinearAlgebra.mul!(dest, idecomp::InequalityDecomp, v, a, b)
	n = length(idecomp.Dx)
	Jct = idecomp.Jct
	m = size(Jct, 2)
	Dx = idecomp.Dx
	Dy = idecomp.Dy
	S = idecomp.S

	# dest[1:n] = a*Jct*v[n+1:n+m] + b*dest
	mul!(view(dest, 1:n), Jct, view(v, n+1:n+m), a, b)

	# dest += a*[Dx; Dy]*S * v[1:n]
	dest[1:n] .+= a .* Dx .* S .* view(v, 1:n)
	dest[n+1:2*n] .*= b
	dest[n+1:2*n] .+= a .* Dy .* S .* view(v, 1:n)

	return dest
end

# transpose projection
function LinearAlgebra.mul!(dest, ida::InequalityDecompAdjoint, v)
	idecomp = ida.idecomp
	n = length(idecomp.Dx)
	Jct = idecomp.Jct
	m = size(Jct, 2)
	Dx = idecomp.Dx
	Dy = idecomp.Dy
	S = idecomp.S

	# dest[1:n] = [Dx; Dy]*v[1:n]
	dest[1:n] .= S .* Dx .* view(v, 1:n)
	dest[1:n] .+= S .* Dy .* view(v, n+1:2*n)

	# dest[n+1:n+rank] = U'*v
	mul!(view(dest, n+1:n+m), Jct', view(v, 1:n))

	return dest
end

#=
Calculates Lagrange multipliers for projected gradient

INPUTS
Qt∇f - n+m vector of coefficients of projection of negative gradient
idecomp - decomposition
idata - inequality data structure describing constraints

OVERWRITTEN
λ_kkt and λy_kkt with Lagrange multipliers
Qt∇f for work

=#
function calculate_λ_kkt!(λ_kkt, λy_kkt, Qt∇f, idecomp::InequalityDecomp)
	n = length(idecomp.Dx)
	rank = idecomp.rank
	Σ = idecomp.Σ
	m = length(Σ)

	# Σ^{-1}
	@inbounds @simd for j = 1:rank
		Qt∇f[n+j] /= Σ[j]
	end
	@inbounds @simd for j = rank+1:m
		Qt∇f[n+j] = 0.0
	end

	mul!(λ_kkt, idecomp.Vt', view(Qt∇f, n+1:n+m)) # λ = -V Σ^{-1} U' g

	# λy = -S^{-1} R V Σ^{-1}Qt∇f[1:n] + S^{-1} Qt∇f[n+1:n+m]
	mul!(λy_kkt, idecomp.Jct, λ_kkt)
	λy_kkt .*= -1.0 .* idecomp.Dx ./ idecomp.S
	λy_kkt .+= view(Qt∇f, 1:n) ./ idecomp.S

	return λ_kkt, λy_kkt
end