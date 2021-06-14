#= Generates gradient of the objective function with ReverseDiff
using a representative input x0
=#
function generate_gradient(f, x0)
	cfg_f = ReverseDiff.GradientConfig(x0)

	function grad!(g, x)
		ReverseDiff.gradient!(g, f, x, cfg_f)
	end

	return grad!
end

#= Generates gradient of the objective function with ReverseDiff
using a representative input x0 and compiles the tape.

This is useful if there are no branches in the code. If there
are branches, then compiling the tape could result in incorrect
gradients for certain inputs.
=#
function generate_gradient_tape(f, x0)
	cfg_f = ReverseDiff.GradientConfig(x0)
	tape_f = ReverseDiff.GradientTape(f, x0, cfg_f)
	ctape_f = ReverseDiff.compile(tape_f)

	function grad!(g, x)
		ReverseDiff.gradient!(g, ctape_f, x)
	end

	return grad!
end

#= Generates Jacobian of the constraints with Reversediff
using a representative input x0 and output size m
=#
function generate_jacobian(c!, x0, m::Int)
	tmp_m = similar(x0, m)
	cfg_c = ReverseDiff.JacobianConfig(tmp_m, x0)

	function jac!(Jc, cval, x)
		ReverseDiff.jacobian!(Jc, c!, cval, x, cfg_c)
	end

	return jac!
end

#= Generates Jacobian of the constraints with Reversediff
using a representative input x0 and output size m and compiles
a tape.

This is useful if there are no branches in the code. If there
are branches, then compiling the tape could result in incorrect
gradients for certain inputs.
=#
function generate_jacobian_tape(c!, x0, m::Int)
	tmp_m = similar(x0, m)
	cfg_c = ReverseDiff.JacobianConfig(tmp_m, x0)
	tape_c = ReverseDiff.JacobianTape(c!, tmp_m, x0, cfg_c)
	ctape_c = ReverseDiff.compile(tape_c)

	function jac!(Jc, cval, x)
		ReverseDiff.jacobian!(Jc, ctape_c, x)
		ReverseDiff.extract_result_value!(cval, ReverseDiff.output_hook(ctape_c))
	end

	return jac!
end

#= Generates function to calculate the application of the Hessian
of the Lagrangian on a vector with a representative input of x0
=#
function generate_hess_lag_vec(grad_dual!, jac_dual!, x0, m::Int)
	n = length(x0)

	duals_n = zeros(Dual{nothing,Float64,1}, n)
	duals_m = zeros(Dual{nothing,Float64,1}, m)
	duals_jac = zeros(Dual{nothing,Float64,1}, m, n)
	x_dual = zeros(Dual{nothing,Float64,1}, n)

	function hess_lag_vec!(dest, src, x, λ)
		a = Dual{nothing}(0.0, 1.0)
		x_dual .= x .+ a.*src

		# calculate dual objective function gradient
		grad_dual!(duals_n, x_dual)

		# extract Hessian-vector products
		@inbounds for i = 1:n
			dest[i] = duals_n[i].partials[1]
		end

		# calculate dual constraint Jacobian
		jac_dual!(duals_jac, duals_m, x_dual)

		@inbounds for j = 1:n
			dest[j] += sum(duals_jac[i,j].partials[1]*λ[i] for i in 1:m)
		end

		return dest
	end

	return hess_lag_vec!
end