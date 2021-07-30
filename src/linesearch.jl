struct ArmijoWork
	xtilde::Vector{Float64}
end

ArmijoWork(n::Int) = ArmijoWork(Array{Float64}(undef, n))

struct ExactLinesearchWork
	tmp_n1::Vector{Float64}
	tmp_n2::Vector{Float64}
	tmp_n3::Vector{Float64}
	tmp_n4::Vector{Float64}
end

ExactLinesearchWork(n::Int) = ExactLinesearchWork([Array{Float64}(undef, n) for i in 1:4]...)


#=
Performs Armijo backtracking linesearch

OUTPUT
flag - any flags from the retraction or Armijo error
iter1 - from retraction
iter2 - from retraction
newf - new function value at xnew
f_diff - difference in function value from previous iterate
step_diff - norm of x difference from pervious iterate

OVERWRITES
xnew - with new x

=#
function armijo!(xnew, x, n, d, g, f, fval, retract_method, cval, c!, param::LFPSQPParams, work::ArmijoWork)
	f_diff = Inf
	step_diff = Inf
	α = param.α
	flag = 0
	tot_iter1 = 0
	tot_iter2 = 0
	newf = 0.0

	ar_dot = dot(d, g)

	# unpack
	xtilde = work.xtilde
	step = xtilde				# reuse xtilde for step size calculation at end


	while step_diff > param.ϵ_x
		xtilde .= x .+ α.*d

		# perform retraction
		flag, iter1, iter2 = retract!(cval, xnew, c!, xtilde, x, retract_method)
		tot_iter1 += iter1
		tot_iter2 += iter2

		# if there's an issue, shrink the step size
		if flag > 0
			α *= param.s
			continue
		end

		# calculate new function values
		step .= xnew .- x
		newf = f(xnew)

		step_diff = norm(view(step, 1:n))	# only use first n values for step_diff
		f_diff = abs(newf - fval)

		# break conditions
		if param.disable_linesearch
			break
		end

		# Armijo-Goldstein condition
		if (newf - fval) <= param.σ * α * ar_dot
			break
		end

		α *= param.s

		# to prevent infinite loop
		if α < 1e-100
			flag = 99
			break
		end
	end

	return flag, tot_iter1, tot_iter2, newf, f_diff, step_diff, α
end


#=
Performs "exact" golden-section bisection with upper bounding from McCormick

OUTPUT
flag - any flags from the retraction or Armijo error
iter1 - from retraction
iter2 - from retraction
newf - new function value at xnew
f_diff - difference in function value from previous iterate
step_diff - norm of x difference from pervious iterate

OVERWRITES
xnew - with new x

=#
function exact_linesearch!(xnew, x, n, d, f, fval, retract_method, cval, c!, param::LFPSQPParams, work::ExactLinesearchWork)

	ϕ1 = (3 - sqrt(5))/2
	ϕ2 = (sqrt(5) - 1)/2
	ϕ3 = (sqrt(5) + 1)/2
	Δ = param.α     		# use previous α as step length guess

	flag = 0
	tot_iter1 = 0
	tot_iter2 = 0
	newf = 0.0
	
	f_a = 0.0
	f_b = 0.0
	f_c = 0.0
	f_d = 0.0

	α_a = 0.0
	α_b = 0.0
	α_c = 0.0
	α_d = 0.0

	# unpack work and assign to x_{a,b,c,d} values
	x_a = work.tmp_n1
	x_b = work.tmp_n2
	x_c = work.tmp_n3
	x_d = work.tmp_n4
	step = work.tmp_n1		# reuse tmp_n1 for step size calculation at end

	do_shrinking = true
	flag = 0
	exact_iter_count = 0

	# find the upper bound and maintain rotating list of points
	x_d .= x
	f_d = fval

	# growing
	while true
		exact_iter_count += 1

		swp = x_b	# rotate all of the position vectors
		x_b = x_c
		x_c = x_d
		x_d = swp

		f_b = f_c
		f_c = f_d
		α_b = α_c
		α_c = α_d

		x_d .= x .+ (α_d + Δ).*d

		# perform retraction
		flag, iter1, iter2 = retract!(cval, xnew, c!, x_d, x, retract_method)
		tot_iter1 += iter1
		tot_iter2 += iter2

		x_d .= xnew 	# update x_d with retracted step

		α_d += Δ

		# should break if projection failed or if α > 1.0
		if flag > 0 || α_d > 1.0
			f_d = Inf 		# Inf indicates projection failed
			break
		end

		f_d = f(x_d)

		if f_d > f_c
			break
		end

		do_shrinking = false
		Δ *= ϕ3
	end

	# no feasible upper bound found, so perform shrinking 
	if do_shrinking
		# shrinking

		# assign b with α=0 so that point a is assigned correctly below
		f_b = fval
		α_b = 0.0
		x_b .= x

		f_c = Inf
		α_c = Δ

		swp = x_d	# swap with d since d contains the point for α=Δ already
		x_d = x_c
		x_c = swp

		while true
			exact_iter_count += 1

			swp = x_d	# swapping is now "up" toward d since Δ is getting shrunk
			x_d = x_c
			x_c = swp

			f_d = f_c
			α_d = α_c

			x_c .= x .+ (ϕ1*α_c).*d

			# perform retraction
			flag, iter1, iter2 = retract!(cval, xnew, c!, x_c, x, retract_method)
			tot_iter1 += iter1
			tot_iter2 += iter2

			x_c .= xnew 	# update x_c with retracted step

			α_c *= ϕ1

			# should break if projection failed or if α > 1.0
			if flag > 0 || α_c > 1.0
				f_c = Inf
			else
				f_c = f(x_c)
			end

			if f_c <= fval || α_c < 1e-100
				break
			end
		end
	end

	# assign values from upper bounding procedure and calculate point c
	exact_iter_count += 1
	f_a = f_b
	f_b = f_c
	α_a = α_b
	α_b = α_c

	swp = x_a
	x_a = x_b
	x_b = x_c
	x_c = swp

	α_c = α_a + ϕ2*(α_d - α_a)
	x_c .= x .+ α_c.*d

	# perform retraction
	flag, iter1, iter2 = retract!(cval, xnew, c!, x_c, x, retract_method)
	tot_iter1 += iter1
	tot_iter2 += iter2

	x_c .= xnew 	# update x_c with retracted step

	if flag > 0 || α_c > 1.0
		f_c = Inf
	else
		f_c = f(x_c)
	end

	# ----------- main loop ------------------

	# do golden ratio bisection
	nd = norm(d)
	while (α_c - α_b) > 1e-6*nd
		exact_iter_count += 1

		if f_b < f_c || isinf(f_c)	# shrink interval to the left
			swp = x_d
			x_d = x_c
			x_c = x_b
			x_b = swp
			
			f_d = f_c
			f_c = f_b
			α_d = α_c
			α_c = α_b

			# calculate point b (which can never be infinite)
			α_b = α_a + ϕ1*(α_d - α_a)
			x_b .= x .+ α_b.*d

			# perform retraction
			flag, iter1, iter2 = retract!(cval, xnew, c!, x_b, x, retract_method)
			tot_iter1 += iter1
			tot_iter2 += iter2

			x_b .= xnew 	# update x_b with retracted step

			f_b = f(x_b)
		else 						# shrink interval to the right
			swp = x_a
			x_a = x_b
			x_b = x_c
			x_c = swp
			
			f_a = f_b
			f_b = f_c
			α_a = α_b
			α_b = α_c

			# calcaulte point c
			α_c = α_a + ϕ2*(α_d - α_a)
			x_c .= x .+ α_c.*d

			# perform retraction
			flag, iter1, iter2 = retract!(cval, xnew, c!, x_c, x, retract_method)
			tot_iter1 += iter1
			tot_iter2 += iter2

			x_c .= xnew 	# update x_c with retracted step

			if flag > 0 || α_c > 1.0
				f_c = Inf
			else
				f_c = f(x_c)
			end
		end
	end

	# assign final function, x, and α value
	if f_b < f_c
		xnew .= x_b
		newf = f_b
		α = α_b
	else
		xnew .= x_c
		newf = f_c
		α = α_c
	end

	step .= xnew .- x
	step_diff = norm(view(step, 1:n))	# only use first n values for step_diff
	f_diff = abs(newf - fval)

	return flag, tot_iter1, tot_iter2, newf, f_diff, step_diff, α
end