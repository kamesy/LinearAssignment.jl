"""
    struct LAWorkspace{T}
        I::Vector{UInt32}       # assigned row indices for columns, cost[I[j], j]
        J::Vector{UInt32}       # assigned column indices for rows, cost[i, J[i]]
        u::Vector{T}            # dual row variables
        v::Vector{T}            # dual column variables
        d::Vector{T}            # shortest path lengths
        p::Vector{UInt32}       # predecessor array for shortest path tree
        sr::Vector{Bool}        # scanned rows
        sc::Vector{Bool}        # scanned columns
        free::Vector{UInt32}    # unassigned rows
        feasible::Ref{Bool}     # feasible assignment
    end

    LAWorkspace(cost::AbstractMatrix{T}) -> LAWorkspace{T}
    LAWorkspace(::Type{T}, cost::AbstractMatrix) -> LAWorkspace{T}
    LAWorkspace(::Type{T}, m::Integer, n::Integer) -> LAWorkspace{T}
"""
struct LAWorkspace{T}
    I::Vector{UInt32}           # assigned row indices for columns, cost[I[j], j]
    J::Vector{UInt32}           # assigned column indices for rows, cost[i, J[i]]
    u::Vector{T}                # dual row variables
    v::Vector{T}                # dual column variables
    d::Vector{T}                # shortest path lengths
    p::Vector{UInt32}           # predecessor array for shortest path tree
    sr::Vector{Bool}            # scanned rows
    sc::Vector{Bool}            # scanned columns
    free::Vector{UInt32}        # unassigned rows
    feasible::RefValue{Bool}    # feasible assignment

    function LAWorkspace{T}(
        I::Vector{UInt32},
        J::Vector{UInt32},
        u::Vector{T},
        v::Vector{T},
        d::Vector{T},
        p::Vector{UInt32},
        sr::Vector{Bool},
        sc::Vector{Bool},
        free::Vector{UInt32}
    ) where {T}
        m = length(J)
        n = length(I)
        @assert length(u) == m
        @assert length(v) == n
        @assert length(d) == m
        @assert length(p) == m
        @assert length(sr) == m
        @assert length(sc) == n
        @assert length(free) == m
        new{T}(I, J, u, v, d, p, sr, sc, free, Ref(false))
    end
end

LAWorkspace(C::AbstractMatrix{T}) where {T} =
    LAWorkspace(T, size(C)...)

LAWorkspace(::Type{T}, C::AbstractMatrix) where {T} =
    LAWorkspace(T, size(C)...)

function LAWorkspace(::Type{T}, m::Integer, n::Integer) where {T}
    I  = fill!(Vector{UInt32}(undef, n), 0)
    J  = fill!(Vector{UInt32}(undef, m), 0)
    v  = fill!(Vector{T}(undef, n), 0)
    f  = fill!(Vector{UInt32}(undef, m), 0)
    u  = fill!(Vector{T}(undef, m), 0)
    d  = fill!(Vector{T}(undef, m), typemax(T))
    p  = fill!(Vector{UInt32}(undef, m), 0)
    sr = fill!(Vector{Bool}(undef, m), 0)
    sc = fill!(Vector{Bool}(undef, n), 0)
    return LAWorkspace{T}(I, J, u, v, d, p, sr, sc, f)
end


"""
    isfeasible(L::LAWorkspace) -> Bool

Return feasibility of assignment.
"""
isfeasible(L::LAWorkspace) = L.feasible[]


"""
    linear_assignment(
        cost::AbstractMatrix{T},
        early_exit::Bool = true
    ) -> LAWorkspace{T}

Solve the linear assignment problem using a modified Jonker-Volgenant algorithm [1].

The cost of the assignment can be computed using `compute_cost(L, cost)`.
The assigned row and column indices can be found in LAWorkspace.I,J, respectively.

### Arguments
- `cost::AbstractMatrix{T}`: rectangular cost matrix
- `early_exit::Bool = true`: exit as soon as infeasibility is detected

### Returns
- `LAWorkspace{T}`: struct containing assigned indices, dual vectors, and more.

### References
[1] DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696,
    August 2016, DOI:10.1109/TAES.2016.140952
"""
function linear_assignment(C::AbstractMatrix, early_exit::Bool = true)
    require_one_based_indexing(C)
    L = LAWorkspace(C)
    linear_assignment!(L, C, early_exit, false)
    return L
end


"""
    linear_assignment!(
        L::LAWorkspace{T},
        cost::AbstractMatrix{Tc},
        early_exit::Bool = true,
        reset::Bool = true
    ) -> LAWorkspace{T}

Solve the linear assignment problem using a modified Jonker-Volgenant algorithm [1].

The cost of the assignment can be computed using `compute_cost(L, cost)`.
The assigned row and column indices can be found in LAWorkspace.I,J, respectively.

### Arguments
- `L::LAWorkspace{T}`: workspace struct
- `cost::AbstractMatrix{Tc}`: rectangular cost matrix
- `early_exit::Bool = true`: exit as soon as infeasibility is detected
- `reset::Bool = true`: zero out assigned indices and dual vectors

### Returns
- `LAWorkspace{T}`: struct containing assigned indices, dual vectors, and more.

### References
[1] DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696,
    August 2016, DOI:10.1109/TAES.2016.140952
"""
linear_assignment!(
    ::LAWorkspace{<:Integer},
    ::AbstractMatrix{Tc},
    ::Bool = true,
    ::Bool = true
) where {Tc<:AbstractFloat} = throw(
    ArgumentError("float cost matrix with integer workspace not supported")
)

function linear_assignment!(
    L::LAWorkspace{T},
    C::AbstractMatrix{Tc},
    early_exit::Bool = true,
    reset::Bool = true
) where {T, Tc}
    require_one_based_indexing(C)
    m, n = size(C)
    @assert m <= length(L.J)
    @assert n <= length(L.I)
    I, J = L.I, L.J
    u, v = L.u, L.v
    d, p = L.d, L.p
    sr, sc = L.sr, L.sc
    free, feasible = L.free, L.feasible

    feasible[] = true

    if reset
        reset!(L, m, n)
    end

    @inbounds for j in 1:n
        sc[j] = false
    end

    @inbounds for i in 1:m
        sr[i] = false
    end

    # there seems to be a 10-30% performance boost when we index
    # the cost matrix via pointers pointing at the current column.
    # to unnecessarily remove one addition we start the pointer
    # offset behind the first element:
    # C[i,j] -> C[i + m*(j-1)] -> C[i-m + m*j] -> ptr[m*j]
    inc = UInt(sizeof(Tc)*m)
    cptr = pointer(C, 1-m)

    R32 = Base.oneto(UInt32(m))

    @inbounds for col in 1:n
        for i in R32
            free[i] = i
        end

        for i in 1:m
            p[i] = col
        end

        # unroll the first iteration to skip some if statements
        ptr = cptr + inc*col
        row = shortest_path1!(d, u, v, J, ptr, col, m)

        δ = typemax(T)
        sink = 0

        mf = m
        j = col
        i = row

        while i != 0
            δ = d[i]
            sr[i] = true

            j = Int(J[i])
            if j == 0
                sink = i
                break
            end
            sc[j] = true

            mf -= 1
            for f in row:mf
                free[f] = free[f+1]
            end

            ptr = cptr + inc*j
            vfree = view(free, 1:mf)

            row = shortest_path!(d, p, u, v, vfree, ptr, δ, j)
            i = row == 0 ? 0 : Int(free[row])
        end

        if sink == 0
            feasible[] = false
            early_exit && break

            for j in 1:n
                sc[j] || continue
                sc[j] = false
            end
            for i in 1:m
                sr[i] || continue
                sr[i] = false
            end
            continue
        end

        v[col] += δ

        for j in 1:n
            sc[j] || continue
            sc[j] = false
            v[j] += δ - d[I[j]]
        end

        for i in 1:m
            sr[i] || continue
            sr[i] = false
            u[i] += d[i] - δ
        end

        i1 = sink
        while true
            j1 = p[i1]
            J[i1] = j1
            i1, I[j1] = Int(I[j1]), i1
            j1 == col && break
        end
    end

    return L
end

# first iteration, j = col
function shortest_path1!(
    d::AbstractVector{T},
    u::AbstractVector{T},
    v::AbstractVector{T},
    J::AbstractVector{<:Integer},
    ptr::Ptr{Tc},
    col::Integer,
    m::Integer
) where {T, Tc}
    δ = typemax(T)
    row = 0
    @inbounds vj = v[col]

    @inbounds for i in 1:m
        ci = unsafe_load(ptr, i)
        uv = u[i] + vj
        di = ci - uv
        d[i] = di
        if di < δ || (di == δ && J[i] == 0)
            δ = di
            row = i
        end
    end
    # di == δ could pick an inf cost
    # to avoid a branch inside the loop
    # we'll clean up here
    return δ == typemax(δ) ? 0 : row
end

# first iteration, j = col
function shortest_path1!(
    d::AbstractVector{T},
    u::AbstractVector{T},
    v::AbstractVector{T},
    J::AbstractVector{<:Integer},
    ptr::Ptr{Tc},
    col::Integer,
    m::Integer
) where {T<:Integer, Tc<:Integer}
    δ = typemax(T)
    row = 0
    @inbounds vj = v[col]

    TT = promote_type(T, Tc)

    # (todo) overflow?
    @inbounds for i in 1:m
        ci = unsafe_load(ptr, i)
        uv = u[i] + vj
        di = ci == typemax(Tc) ? typemax(TT) : (ci - uv)
        d[i] = di
        if di < δ || (di == δ && J[i] == 0)
            δ = di
            row = i
        end
    end
    # di == δ could pick an inf cost
    # to avoid a branch inside the loop
    # we'll clean up here
    return δ == typemax(δ) ? 0 : row
end


function shortest_path!(
    d::AbstractVector{T},
    p::AbstractVector{<:Integer},
    u::AbstractVector{T},
    v::AbstractVector{T},
    free::AbstractVector{<:Integer},
    ptr::Ptr{Tc},
    δv::T,
    col::Integer,
) where {T, Tc}
    δ = typemax(T)
    row = 0
    @inbounds δv -= v[col]

    @inbounds for f in eachindex(free)
        i = free[f]
        ci = unsafe_load(ptr, i)
        δuv = u[i] - δv
        dj = ci - δuv
        if dj < d[i]
            d[i] = dj
            p[i] = col
            dj < δ || continue
        end
        if d[i] < δ
            δ = d[i]
            row = f
        end
    end

    return row
end

function shortest_path!(
    d::AbstractVector{T},
    p::AbstractVector{<:Integer},
    u::AbstractVector{T},
    v::AbstractVector{T},
    free::AbstractVector{<:Integer},
    ptr::Ptr{Tc},
    δv::T,
    col::Integer,
) where {T<:Integer, Tc<:Integer}
    δ = typemax(T)
    row = 0
    @inbounds δv -= v[col]

    @inbounds for f in eachindex(free)
        i = free[f]
        ci = unsafe_load(ptr, i)
        δuv = u[i] - δv
        if ci < d[i] + δuv
            dj = ci - δuv
            d[i] = dj
            p[i] = col
            dj < δ || continue
        end
        if d[i] < δ
            δ = d[i]
            row = f
        end
    end

    return row
end

function shortest_path!(
    d::AbstractVector{T},
    p::AbstractVector{<:Integer},
    u::AbstractVector{T},
    v::AbstractVector{T},
    free::AbstractVector{<:Integer},
    ptr::Ptr{Tc},
    δv::T,
    col::Integer,
) where {T<:Signed, Tc<:Signed}
    δ = typemax(T)
    row = 0
    @inbounds δv -= v[col]

    @inbounds for f in eachindex(free)
        i = free[f]
        ci = unsafe_load(ptr, i)
        δuv = u[i] - δv
        dj, bad = sub_with_overflow(promote(ci, δuv)...)
        if !bad && dj < d[i]
            d[i] = dj
            p[i] = col
            dj < δ || continue
        end
        if (!bad && dj < δ) || d[i] < δ
            δ = d[i]
            row = f
        end
    end

    return row
end


#####
##### StaticMatrix
#####

"""
    linear_assignment(
        cost::Union{SMatrix{M, N, T}, MMatrix{M, N, T}},
        early_exit::Bool = true
    ) -> Tuple{
        SVector{N, UInt8},  # assigned row indices for columns, cost[I[j], j]
        SVector{M, UInt8},  # assigned column indices for rows, cost[i, J[i]]
        SVector{M, T},      # dual row variables
        SVector{N, T},      # dual column variables
        Bool                # feasible assignment
    }

Solve the linear assignment problem using a modified Jonker-Volgenant algorithm [1].

The cost of the assignment can be computed using `compute_cost(cost, I)`.

### Arguments
- `cost::Union{SMatrix{M, N, T}, MMatrix{M, N, T}}`: rectangular cost matrix
- `early_exit::Bool = true`: exit as soon as infeasibility is detected

### Returns
- `SVector{N, UInt8}`: assigned row indices for columns
- `SVector{M, UInt8}`: assigned column indices for rows
- `SVector{M, T}`: dual row variables
- `SVector{N, T}`: dual column variables
- `Bool`: is assignment feasible

### References
[1] DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696,
    August 2016, DOI:10.1109/TAES.2016.140952
"""
@generated function linear_assignment(
    C::Union{SMatrix{M, N, T, L}, MMatrix{M, N, T, L}},
    early_exit::Bool = true
) where {M, N, T, L}

    shortest_path1_di = T <: AbstractFloat ?
        :(C[i,j] - vj - u[i]) :
        :(C[i,j] == typemax(T) ? typemax(T) : (C[i,j] - vj - u[i]))

    shortest_path = if T <: AbstractFloat
        quote
            for i in 1:$M
                if !sr[i]
                    di = C[i,j] + δv - u[i]
                    if di < d[i]
                        d[i] = di
                        p[i] = j
                        di < δ || continue
                    end
                    if d[i] < δ
                        δ = d[i]
                        row = i
                    end
                end
            end
        end
    else
        quote
            for i in 1:$M
                if !sr[i]
                    δuv = u[i] - δv
                    if C[i,j] < d[i] + δuv
                        di = C[i,j] - δuv
                        d[i] = di
                        p[i] = j
                        d[i] < δ || continue
                    end
                    if d[i] < δ
                        δ = d[i]
                        row = i
                    end
                end
            end
        end
    end

    quote
        u  = @MVector zeros(T, M)
        v  = @MVector zeros(T, N)
        d  = @MVector zeros(T, M)
        I  = @MVector zeros(UInt8, N)
        J  = @MVector zeros(UInt8, M)
        p  = @MVector zeros(UInt8, M)
        sr = @MVector zeros(Bool, M)
        sc = @MVector zeros(Bool, N)

        feasible = true

        @inbounds for col in 1:N
            fill!(p, col)

            δ = typemax(T)
            row = 0

            j = col
            vj = v[j]

            for i in 1:M
                di = $shortest_path1_di
                d[i] = di
                if di < δ || (di == δ && J[i] == 0)
                    δ = di
                    row = i
                end
            end

            sink = 0

            while δ != typemax(T)
                sr[row] = true

                j = Int(J[row])

                if j == 0
                    sink = row
                    break
                end

                sc[j] = true

                δv = δ - v[j]
                δ = typemax(T)
                row = 0

                $shortest_path
            end

            if sink == 0
                feasible = false
                early_exit && break
                for j in 1:N
                    sc[j] || continue
                    sc[j] = false
                end
                for i in 1:M
                    sr[i] || continue
                    sr[i] = false
                end
                continue
            end

            v[col] += δ

            for j in 1:N
                sc[j] || continue
                sc[j] = false
                i = I[j]
                v[j] += δ - d[i]
            end

            for i in 1:M
                sr[i] || continue
                sr[i] = false
                u[i] += d[i] - δ
            end

            i1 = sink
            while true
                j1 = p[i1]
                J[i1] = j1
                i1, I[j1] = I[j1], i1
                j1 == col && break
            end
        end

        return SVector(I), SVector(J), SVector(u), SVector(v), feasible
    end
end


#####
##### Misc
#####

"""
    compute_cost(L::LAWorkspace, C::AbstractMatrix{T}) -> T
    compute_cost(C::AbstractMatrix{T}, I::AbstractVector{<:Integer}) -> T

Return the cost of the assignment.
"""
compute_cost(L::LAWorkspace, C::AbstractMatrix) = compute_cost(C, L.I)

function compute_cost(
    C::AbstractMatrix{T},
    I::AbstractVector{<:Integer}
) where {T}
    require_one_based_indexing(C, I)
    m, n = size(C)
    @assert n <= length(I)

    cost = zero(T <: AbstractFloat ? promote_type(Float64, T) : T)
    @inbounds for j in 1:n
        i = I[j]
        if I != 0
            cost += C[i,j]
        end
    end

    return convert(T, cost)
end


"""
    reset!(::LAWorkspace{T}) -> LAWorkspace{T}
    reset!(::LAWorkspace{T}, C::AbstractMatrix) -> LAWorkspace{T}
    reset!(::LAWorkspace{T}, m::Integer, n::Integer) -> LAWorkspace{T}

Zero out assigned indices and dual variables.
"""
reset!(L::LAWorkspace) = reset!(L, length(L.x), length(L.y))
reset!(L::LAWorkspace, C::AbstractMatrix) = reset!(L, size(C)...)

function reset!(L::LAWorkspace{T}, m::Integer, n::Integer) where {T}
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
