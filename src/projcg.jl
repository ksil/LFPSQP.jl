struct ProjCGWork{T}
    r::Vector{T}
    g::Vector{T}
    d::Vector{T}
    rp::Vector{T}
    gp::Vector{T}
    Ad::Vector{T}
    Utr::Vector{T}
end

ProjCGWork(n::Int, m::Int) = ProjCGWork([Array{Float64}(undef, n) for i in 1:6]..., Array{Float64}(undef, m))


#=
PROJCG does projected conjugate gradient with orthonormal projection matrix
U and PSD matrix A. That is, the following system is solved:

[ A  U ] [ x ] = [ b ]
[ U' 0 ] [ λ ]   [ c ]

Returns a direction of negative curvature of A if found, storing the direction
in x and setting λ to all NaNs.

Usage:
  [x, lambda, resvec, Uvec, dvec] = projcg(A, b, U, c, tol, maxit)

Inputs:
A, b, U, c - as defined above
tol [1e-6] - absolute tolerance for convergence of 2-norm of residual on the
  constraint manifold
maxit [size(A,1) - size(U,2)] - maximum number of iterations if tol is not met

Overwrites:
x, λ - solution

Output:
i - number of iterations taken
nr - final tolerance achieved for projected residual, (I - U*U')*(A*x - b)
=#
function projcg!(x, λ, A, U, b, c;
    tol::Float64=1e-6, maxit::Int=length(b)+length(c), work::ProjCGWork{T}=ProjCGWork(length(b), length(c))) where T

n = length(b)
m = length(c)

# work variables
r = view(work.r, 1:n)
g = view(work.g, 1:n)
d = view(work.d, 1:n)
rp = view(work.rp, 1:n)
gp = view(work.gp, 1:n)
Ad = view(work.Ad, 1:n)
Utr = view(work.Utr, 1:m)

mul!(x, U, c)                       # satisfies U' x = c
r .= b
mul!(r, A, x, 1.0, -1.0)            # r = Ax - b
g .= r
mul!(Utr, U', r)
mul!(g, U, Utr, -1.0, 1.0)          # g = r - U*(U'*r);
r .= g
d .= -1.0.*g                        # multiplication avoids allocation

nr0 = norm(g);

# resvec = [nr0; zeros(maxit, 1)];

i = 0
nr = Inf

while i < min(maxit, n+m)
    i += 1

    mul!(Ad, A, d)                  # Ad = A*d
    dAd = dot(d, Ad);
    
    if dAd <= 0
        # resvec = resvec(1:i);
        x .= d / norm(d)
        λ .= NaN
        return i, Inf
    end
    
    rg = dot(r, g);
    
    # rg cannot be negative
    if rg <= 0
        break
    end
    
    α = rg / dAd
    x .+= α.*d
    rp .= r .+ α.*Ad

    gp .= rp
    mul!(Utr, U', rp)
    mul!(gp, U, Utr, -1.0, 1.0)      # gp = rp - U*(U'*rp);
    β = dot(rp, gp) / rg
    d .= β.*d .- gp
    g .= gp
    r .= gp
    
    nr = norm(g);
    # resvec(i+1) = nr;
    

    if nr < tol
        # resvec = resvec(1:i+1);
        
        break
    end
end

# solve for Lagrange multipliers
r .= b
mul!(r, A, x, -1.0, 1.0)            # r = b - A*x

mul!(λ, U', r);          # λ = U'*(b - A*x)

return i, nr
end