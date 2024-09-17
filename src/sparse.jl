#####
##### SparseMatrixCSC
#####

"""
    struct LAWorkspaceCSC{Tv}
        I::Vector{UInt32}           # assigned row indices for columns, cost[I[j], j]
        J::Vector{UInt32}           # assigned column indices for rows, cost[i, J[i]]
        u::Vector{Tv}               # dual row variables
        v::Vector{Tv}               # dual column variables
        d::Vector{Tv}               # shortest path lengths
        p::Vector{UInt32}           # predecessor array for shortest path tree
        srv::Vector{Bool}           # scanned rows visited
        sri::Vector{UInt32}         # scanned rows indices
        scv::Vector{Bool}           # scanned cols visited
        sci::Vector{UInt32}         # scanned cols indices
        sdv::Vector{Bool}           # set path lengths visited
        sdi::Vector{UInt32}         # set path lengths indices
        qdi::Vector{UInt32}         # set path lengths indices not scanned
        qdj::Vector{UInt32}         # set path lengths indices not scanned reverse
        feasible::RefValue{Bool}    # feasible assignment
    end

    LAWorkspaceCSC(cost::SparseMatrixCSC{Tv}) -> LAWorkspaceCSC{Tv}
    LAWorkspaceCSC(::Type{Tv}, cost::SparseMatrixCSC) -> LAWorkspaceCSC{Tv}
    LAWorkspaceCSC(::Type{Tv}, m::Integer, n::Integer) -> LAWorkspaceCSC{Tv}
"""
struct LAWorkspaceCSC{Tv}
    I::Vector{UInt32}           # assigned row indices for columns, cost[I[j], j]
    J::Vector{UInt32}           # assigned column indices for rows, cost[i, J[i]]
    u::Vector{Tv}               # dual row variables
    v::Vector{Tv}               # dual column variables
    d::Vector{Tv}               # shortest path lengths
    p::Vector{UInt32}           # predecessor array for shortest path tree
    srv::Vector{Bool}           # scanned rows visited
    sri::Vector{UInt32}         # scanned rows indices
    scv::Vector{Bool}           # scanned cols visited
    sci::Vector{UInt32}         # scanned cols indices
    sdv::Vector{Bool}           # set path lengths visited
    sdi::Vector{UInt32}         # set path lengths indices
    qdi::Vector{UInt32}         # set path lengths indices not scanned
    qdj::Vector{UInt32}         # set path lengths indices not scanned reverse
    feasible::RefValue{Bool}    # feasible assignment

    function LAWorkspaceCSC{Tv}(
        I::Vector{UInt32},
        J::Vector{UInt32},
        u::Vector{Tv},
        v::Vector{Tv},
        d::Vector{Tv},
        p::Vector{UInt32},
        srv::Vector{Bool},
        sri::Vector{UInt32},
        scv::Vector{Bool},
        sci::Vector{UInt32},
        sdv::Vector{Bool},
        sdi::Vector{UInt32},
        qdi::Vector{UInt32},
        qdj::Vector{UInt32},
    ) where {Tv}
        m = length(J)
        n = length(I)
        @assert length(u) == m
        @assert length(v) == n
        @assert length(d) == m
        @assert length(p) == m
        @assert length(srv) == m
        @assert length(sri) == m
        @assert length(scv) == n
        @assert length(sci) == n
        @assert length(sdv) == m
        @assert length(sdi) == m
        @assert length(qdi) == m
        @assert length(qdj) == m
        new{Tv}(I, J, u, v, d, p, srv, sri, scv, sci, sdv, sdi, qdi, qdj, Ref(false))
    end
end

LAWorkspaceCSC(C::SparseMatrixCSC{Tv}) where {Tv} =
    LAWorkspaceCSC(Tv, size(C)...)

LAWorkspaceCSC(::Type{Tv}, C::SparseMatrixCSC) where {Tv} =
    LAWorkspaceCSC(Tv, size(C)...)

function LAWorkspaceCSC(::Type{Tv}, m::Integer, n::Integer) where {Tv}
    I   = fill!(Vector{UInt32}(undef, n), 0)
    J   = fill!(Vector{UInt32}(undef, m), 0)
    v   = fill!(Vector{Tv}(undef, n), 0)
    srv = fill!(Vector{Bool}(undef, m), 0)
    u   = fill!(Vector{Tv}(undef, m), 0)
    d   = fill!(Vector{Tv}(undef, m), typemax(Tv))
    p   = fill!(Vector{UInt32}(undef, m), 0)
    sdv = fill!(Vector{Bool}(undef, m), 0)
    sdi = fill!(Vector{UInt32}(undef, m), 0)
    qdi = fill!(Vector{UInt32}(undef, m), 0)
    qdj = fill!(Vector{UInt32}(undef, m), 0)
    sri = fill!(Vector{UInt32}(undef, m), 0)
    sci = fill!(Vector{UInt32}(undef, n), 0)
    scv = fill!(Vector{Bool}(undef, n), 0)
    return LAWorkspaceCSC{Tv}(
        I, J,
        u, v,
        d, p,
        srv, sri,
        scv, sci,
        sdv, sdi,
        qdi, qdj
    )
end


"""
    isfeasible(L::LAWorkspaceCSC) -> Bool

Return feasibility of assignment.
"""
isfeasible(L::LAWorkspaceCSC) = L.feasible[]


"""
    linear_assignment(
        cost::SparseMatrixCSC{Tv},
        early_exit::Bool = true
    ) -> LAWorkspaceCSC{Tv}

Solve the sparse linear assignment problem using a modified Jonker-Volgenant algorithm [1].

The cost of the assignment can be computed using `compute_cost(L, cost)`.
The assigned row and column indices can be found in LAWorkspaceCSC.I,J, respectively.

### Arguments
- `cost::SparseMatrixCSC{Tv}`: sparse rectangular cost matrix
- `early_exit::Bool = true`: exit as soon as infeasibility is detected

### Returns
- `LAWorkspaceCSC{Tv}`: struct containing assigned indices, dual vectors, and more.

### References
[1] DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696,
    August 2016, DOI:10.1109/TAES.2016.140952
"""
function linear_assignment(C::SparseMatrixCSC, early_exit::Bool = true)
    L = LAWorkspaceCSC(C)
    linear_assignment!(L, C, early_exit, false)
    return L
end


"""
    linear_assignment!(
        L::LAWorkspaceCSC{Tv},
        cost::SparseMatrixCSC{Tc},
        early_exit::Bool = true,
        reset::Bool = true
    ) -> LAWorkspaceCSC{Tv}

    linear_assignment!(
        L::LAWorkspaceCSC{Tv},
        m::Integer,
        n::Integer,
        colptr::AbstractVector{<:Integer},
        rowval::AbstractVector{<:Integer},
        nzval::AbstractVector{Tc},
        early_exit::Bool = true,
        reset::Bool = true
    ) -> LAWorkspaceCSC{Tv}

Solve the sparse linear assignment problem using a modified Jonker-Volgenant algorithm [1].

The cost of the assignment can be computed using `compute_cost(L, cost)`,
`compute_cost(L, m, n, colptr, rowval, nzval)`.
The assigned row and column indices can be found in LAWorkspaceCSC.I,J, respectively.

### Arguments
- `L::LAWorkspaceCSC{Tv}`: workspace struct
- `cost::SparseMatrixCSC{Tc}`: sparse rectangular cost matrix
- `early_exit::Bool = true`: exit as soon as infeasibility is detected
- `reset::Bool = true`: zero out assigned indices and dual vectors

### Returns
- `LAWorkspaceCSC{T}`: struct containing assigned indices, dual vectors, and more.

### References
[1] DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696,
    August 2016, DOI:10.1109/TAES.2016.140952
"""
function linear_assignment!(
    L::LAWorkspaceCSC,
    C::SparseMatrixCSC,
    early_exit::Bool = true,
    reset::Bool = true
)
    m, n = size(C)
    colptr = getcolptr(C)
    rowval = rowvals(C)
    nzval  = nonzeros(C)
    linear_assignment!(L, m, n, colptr, rowval, nzval, early_exit, reset)
    return L
end

linear_assignment!(
    ::LAWorkspaceCSC{<:Integer},
    ::Integer,
    ::Integer,
    ::AbstractVector{<:Integer},
    ::AbstractVector{<:Integer},
    ::AbstractVector{Tc},
    ::Bool = true,
    ::Bool = true
) where {Tc<:AbstractFloat} = throw(
    ArgumentError("float cost matrix with integer workspace not supported")
)

function linear_assignment!(
    L::LAWorkspaceCSC{Tv},
    m::Integer,
    n::Integer,
    colptr::AbstractVector{<:Integer},
    rowval::AbstractVector{<:Integer},
    nzval::AbstractVector{Tc},
    early_exit::Bool = true,
    reset::Bool = true
) where {Tv, Tc}
    require_one_based_indexing(colptr, rowval, nzval)
    @assert m <= length(L.J)
    @assert n <= length(L.I)
    @assert n <= length(colptr)
    @assert colptr[n]-1 <= length(rowval)
    @assert colptr[n]-1 <= length(nzval)

    I, J = L.I, L.J
    u, v = L.u, L.v
    d, p = L.d, L.p
    srv, sri = L.srv, L.sri
    scv, sci = L.scv, L.sci
    sdv, sdi = L.sdv, L.sdi
    qdi, qdj = L.qdi, L.qdj
    feasible = L.feasible

    feasible[] = true

    if reset
        reset!(L, m, n)
    end

    @inbounds for j in 1:n
        scv[j] = false
    end

    @inbounds for i in 1:m
        srv[i] = false
    end

    @inbounds for i in 1:m
        sdv[i] = false
    end

    @inbounds for i in 1:m
        p[i] = 0
    end

    @inbounds for i in 1:m
        d[i] = typemax(Tv)
    end

    @inbounds for col in 1:n
        nsr = 0
        nsc = 0
        nsd = 0
        nqd = 0

        δ = typemax(Tv)
        j = UInt32(col)
        # the integer variables flip between Int and UInt32 since
        # the index arrays are UInt32s. If we try fixing the type to
        # either this entire thing slows down a lot?
        row = 0

        vj = v[col]
        for s in colptr[col]:colptr[col+1]-1
            i = rowval[s]
            if Tv <: Integer && Tc <: Integer
                ci = nzval[s]
                uv = u[i] + vj
                di = ci == typemax(Tc) ? typemax(promote_type(Tv, Tc)) : (ci - uv)
            else
                ci = nzval[s]
                uv = u[i] + vj
                di = ci - uv
            end
            d[i] = di
            if di < δ || (di == δ && J[i] == 0)
                δ = di
                row = i
            end
            p[i] = j
            sdv[i] = true
            sdi[nsd += 1] = i
            qdi[nqd += 1] = i
            qdj[i] = nqd
        end
        # di == δ could pick an inf cost
        # to avoid a branch inside the loop
        # we'll clean up here
        row = δ == typemax(δ) ? 0 : row

        sink = 0

        while row != 0
            if !srv[row]
                srv[row] = true
                sri[nsr += 1] = row
            end

            j = J[row]

            if j == 0
                sink = Int(row)
                break
            end

            if !scv[j]
                scv[j] = true
                sci[nsc += 1] = j
            end

            nqd -= 1
            for s in qdj[row]:UInt32(nqd)
                i = qdi[s+1]
                qdj[i] = s
                qdi[s] = i
            end

            δv = δ - v[j]
            δ = typemax(Tv)
            row = 0

            for s in 1:nqd
                i = qdi[s]
                if d[i] < δ
                    δ = d[i]
                    row = i
                end
            end

            # negligible performance boost but since
            # we are using pointers in the dense version
            # we might as well use them here too
            s1 = colptr[j]
            ns = colptr[j+1] - s1
            ptrr = pointer(rowval, s1)
            ptrz = pointer(nzval, s1)

            for s in 1:ns
                if Tv <: Integer && Tv <: Integer
                    i = unsafe_load(ptrr, s)
                    ci = unsafe_load(ptrz, s)
                    δuv = u[i] - δv
                    srv[i] && continue
                    if ci < d[i] + δuv
                        dj = ci - δuv
                        d[i] = dj
                        p[i] = j
                        if !sdv[i]
                            sdv[i] = true
                            sdi[nsd += 1] = i
                            qdi[nqd += 1] = i
                            qdj[i] = nqd
                        end
                        dj < δ || continue
                    end
                    if d[i] < δ
                        δ = d[i]
                        row = UInt32(i)
                    end
                else
                    i = unsafe_load(ptrr, s)
                    ci = unsafe_load(ptrz, s)
                    δuv = u[i] - δv
                    dj = ci - δuv
                    srv[i] && continue
                    if dj < d[i]
                        d[i] = dj
                        p[i] = j
                        if !sdv[i]
                            sdv[i] = true
                            sdi[nsd += 1] = i
                            qdi[nqd += 1] = i
                            qdj[i] = nqd
                        end
                        dj < δ || continue
                    end
                    if d[i] < δ
                        δ = d[i]
                        row = i
                    end
                end
            end
        end

        if sink == 0
            feasible[] = false
            early_exit && break

            for s in 1:nsc
                j = sci[s]
                scv[j] = false
            end
            for s in 1:nsr
                i = sri[s]
                srv[i] = false
            end
            for s in 1:nsd
                i = sdi[s]
                p[i] = 0
                d[i] = typemax(Tv)
            end
            for s in 1:nsd
                sdv[sdi[s]] = false
            end
            continue
        end

        v[col] += δ

        for s in 1:nsc
            j = sci[s]
            v[j] += δ - d[I[j]]
            scv[j] = false
        end

        for s in 1:nsr
            i = sri[s]
            u[i] += d[i] - δ
            srv[i] = false
        end

        i1 = sink
        while true
            j1 = p[i1]
            J[i1] = j1
            i1, I[j1] = Int(I[j1]), i1
            j1 == col && break
        end

        for s in 1:nsd
            i = sdi[s]
            p[i] = 0
            d[i] = typemax(Tv)
        end

        for s in 1:nsd
            sdv[sdi[s]] = false
        end
    end

    return L
end


#####
##### Misc
#####

"""
    compute_cost(L::LAWorkspaceCSC, C::SparseMatrixCSC{Tv}) -> Tv

    compute_cost(
        L::LAWorkspaceCSC,
        m::Integer,
        n::Integer,
        colptr::AbstractVector{<:Integer},
        rowval::AbstractVector{<:Integer},
        nzval::AbstractVector{Tv},
    ) -> Tv

    compute_cost(
        m::Integer,
        n::Integer,
        colptr::AbstractVector{<:Integer},
        rowval::AbstractVector{<:Integer},
        nzval::AbstractVector{Tv},
        I::AbstractVector{<:Integer}
    ) -> Tv

Return the cost of the assignment.
"""
compute_cost(L::LAWorkspaceCSC, C::SparseMatrixCSC) =
    compute_cost(L, size(C)..., getcolptr(C), rowvals(C), nonzeros(C))

compute_cost(
    L::LAWorkspaceCSC,
    m::Integer,
    n::Integer,
    colptr::AbstractVector{<:Integer},
    rowval::AbstractVector{<:Integer},
    nzval::AbstractVector{Tv}
) where {Tv} = compute_cost(m, n, colptr, rowval, nzval, L.I)

function compute_cost(
    m::Integer,
    n::Integer,
    colptr::AbstractVector{<:Integer},
    rowval::AbstractVector{<:Integer},
    nzval::AbstractVector{Tv},
    I::AbstractVector{<:Integer}
) where {Tv}
    require_one_based_indexing(colptr, rowval, nzval, I)
    @assert n <= length(colptr)
    @assert n <= length(I)
    @assert colptr[n+1]-1 <= length(rowval)
    @assert colptr[n+1]-1 <= length(nzval)

    cost = zero(Tv <: AbstractFloat ? promote_type(Float64, Tv) : Tv)
    @inbounds for j in 1:n
        i = I[j]
        for s in colptr[j]:colptr[j+1]-1
            if rowval[s] == i
                cost += nzval[s]
                break
            end
        end
    end

    return convert(Tv, cost)
end


"""
    reset!(::LAWorkspaceCSC{T}) -> LAWorkspaceCSC{T}
    reset!(::LAWorkspaceCSC{T}, C::SparseMatrixCSC) -> LAWorkspaceCSC{T}
    reset!(::LAWorkspaceCSC{T}, m::Integer, n::Integer) -> LAWorkspaceCSC{T}

Zero out assigned indices and dual variables.
"""
reset!(L::LAWorkspaceCSC) = reset!(L, length(L.x), length(L.y))
reset!(L::LAWorkspaceCSC, C::SparseMatrixCSC) = reset!(L, size(C)...)

function reset!(L::LAWorkspaceCSC{T}, m::Integer, n::Integer) where {T}
    I, J, u, v = L.I, L.J, L.u, L.v
    @assert m <= length(J)
    @assert n <= length(I)

    @inbounds for j in 1:n
        I[j] = zero(UInt32)
    end

    @inbounds for i in 1:m
        J[i] = zero(UInt32)
    end

    @inbounds for i in 1:m
        u[i] = zero(T)
    end

    @inbounds for j in 1:n
        v[j] = zero(T)
    end

    return L
end
