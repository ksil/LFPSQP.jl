using LinearAlgebra

const liblapack = Base.liblapack_name
const libblas = Base.libblas_name
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra: BlasFloat, BlasInt

function ksvd!(A, U, S, VT, work, calc_work = false)
    m, n   = size(A)
    minmn  = min(m, n)
    lwork  = BlasInt(length(work))
    info   = Ref{BlasInt}()
    
    jobu = 'S'
    jobvt = 'S'
    
    # calculate work
    if calc_work
        lwork = BlasInt(-1)
    end
    
    ccall((@blasfunc(dgesvd_), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
                       Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
                       Ref{BlasInt}, Ptr{BlasInt}),
                      jobu, jobvt, m, n, A, m, S, U, m, VT, n,
                      work, lwork, info)
        
    if calc_work
        lwork = BlasInt(real(work[1]))
        resize!(work, lwork)
    end
end

function kgemv!(tA, rank, alpha, A, x, beta, y)
    ccall((@blasfunc(dgemv_), libblas), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{Float64},
                 Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
                 Ref{Float64}, Ptr{Float64}, Ref{BlasInt}),
                 tA, size(A,1), rank, alpha,
                 A, size(A,1), x, 1,
                 beta, y, 1)
end

# adapted from http://pi.math.cornell.edu/~web6140/TopTenAlgorithms/QRUpdate.html
# w is a work vector
function qrupdate!(Q, R, u, v, w)
    # Compute the QR factorization of Q*R + u*v': 
    
    # Note that Q*R + u*v' = Q*(R + w*v') with w = Q'*u:
    mul!(w, Q', u)
    n = size(Q, 1)
    
    # Convert R+w*v' into upper-hessenberg form using n-1 Givens rotations:
    for k = n-1:-1:1
        c, s, r = LinearAlgebra.givensAlgorithm(w[k], w[k+1])
        w[k+1] = 0.; w[k] = r

        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] + s*R[k+1,j]
            R[k+1,j] = -s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow
            newcol = c*Q[j,k] + s*Q[j,k+1]
            Q[j,k+1] = -s*Q[j,k] + c*Q[j,k+1]
            Q[j,k] = newcol
        end
    end

    # R <- R + w*v' is now upper-hessenberg:
    R[1,:] .+= w[1].*v 
    
    # Convert R from upper-hessenberg form to upper-triangular form using n-1 Givens rotations:
    for k = 1:n-1
        c, s, r = LinearAlgebra.givensAlgorithm(R[k,k], R[k+1,k])

        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] + s*R[k+1,j]
            R[k+1,j] = -s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow
            newcol = c*Q[j,k] + s*Q[j,k+1]
            Q[j,k+1] = -s*Q[j,k] + c*Q[j,k+1]
            Q[j,k] = newcol
        end
    end
end