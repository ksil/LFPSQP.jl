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
