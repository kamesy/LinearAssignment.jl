using LinearAssignment
using Test

using Aqua
using SparseArrays
using StaticArrays


const REALS = (Float64, Float32, Int64, Int32, Int16, UInt64, UInt32, UInt16)

#####
##### Tests are translated from Google's or-tools:
##### https://github.com/google/or-tools/blob/v9.10/ortools/graph/linear_assignment_test.cc
#####

function machol_wien(::Type{T}, m::Integer, n::Integer = m) where {T}
    # Robert E. Machol and Michael Wien, "Errata: A Hard Assignment Problem",
    # Operations Research, vol. 25, p. 364, 1977.
    # http://www.jstor.org/stable/169842
    C = zeros(T, m, n)

    I = zeros(Int, m*n)
    J = zeros(Int, m*n)
    V = zeros(T, m*n)

    l = 1
    for j in axes(C, 2)
        for i in axes(C, 1)
            v = (i-1) * (j-1)
            C[i,j] = v
            V[l] = v
            I[l] = i
            J[l] = j
            l += 1
        end
    end

    S = sparse(I, J, V)

    return S, C
end

const TEST_PROBLEM = Dict(
    :C => [
        90 76 75 80
        35 85 55 65
        125 95 90 105
        45 110 95 115
    ],

    :S => sparse([
        0 76 75 80
        0 85 55 65
        125 95 90 0
        45 110 0 115
    ]),

    :cost => 80 + 55 + 95 + 45,
    :I => [4, 3, 2, 1],
    :J => [4, 3, 2, 1],
)


# insert into sparsematrix without sorting
function add_arc!(
    C::SparseMatrixCSC{T},
    tail::Integer,
    head::Integer,
    cost::Real
) where {T}
    m, n = size(C)
    i = tail + 1
    j = head - m + 1
    (i < 1 || j < 1 || i > m || j > n) && throw(BoundsError(C, (i, j)))

    colptr = SparseArrays.getcolptr(C)
    rowval = rowvals(C)
    nzval  = nonzeros(C)

    for I in colptr[j]:colptr[j+1]-1
        if rowval[I] == i
            nzval[I] = convert(T, cost)
            return nothing
        end
    end

    I = colptr[j+1]
    insert!(rowval, I, i)
    insert!(nzval, I, cost)
    for s in j+1:n+1
        colptr[s] += 1
    end

    return nothing
end

# replace zeros with infs
function _sparse2full(C::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    M = fill!(Matrix{Tv}(undef, size(C)), typemax(Tv))
    I, J, V = findnz(C)
    for i in eachindex(I)
        M[I[i],J[i]] = V[i]
    end
    return M
end


@testset "LinearAssignment.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LinearAssignment)
    end

    @testset "Interface, Types, Allocations - $T" for T in REALS
        C = T.(TEST_PROBLEM[:C])
        D = sparse(C)

        S = T.(TEST_PROBLEM[:S])
        U = _sparse2full(S)

        cost = T(TEST_PROBLEM[:cost])
        I = TEST_PROBLEM[:I]
        J = TEST_PROBLEM[:J]

        LC = LAWorkspace(C)
        linear_assignment!(LC, C, true, false)
        @test isfeasible(LC)
        @test compute_cost(LC, C) == cost
        @test LC.I == I
        @test LC.J == J

        LD = LAWorkspaceCSC(D)
        linear_assignment!(LD, D, true, false)
        @test isfeasible(LD)
        @test compute_cost(LD, D) == cost
        @test LD.I == I
        @test LD.J == J

        LS = LAWorkspaceCSC(S)
        linear_assignment!(LS, S, true, false)
        @test isfeasible(LS)
        @test compute_cost(LS, S) == cost
        @test LS.I == I
        @test LS.J == J

        LU = LAWorkspace(U)
        linear_assignment!(LU, U, true, false)
        @test isfeasible(LU)
        @test compute_cost(LU, U) == cost
        @test LU.I == I
        @test LU.J == J

        @testset "Tv: $Tv" for Tv in REALS
            if T <: AbstractFloat && Tv <: Integer
                L = LAWorkspace(Tv, C)
                @test_throws ArgumentError linear_assignment!(L, C, false, false)

                L = LAWorkspace(Tv, D)
                @test_throws ArgumentError linear_assignment!(L, D, false, false)

                L = LAWorkspace(Tv, S)
                @test_throws ArgumentError linear_assignment!(L, S, false, false)

                L = LAWorkspace(Tv, S)
                @test_throws ArgumentError linear_assignment!(L, U, false, false)

            elseif T <: Integer && typemax(T) > typemax(Tv)
                # TODO: can throw inexact errors on dual update
                # currently not supporting this
                continue

            else
                continue
                L = LAWorkspace(Tv, C)
                linear_assignment!(L, C, false, false)
                @test isfeasible(L)
                @test compute_cost(L, C) == cost

                if Tv === T
                    @test typeof(L) === typeof(LC)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LC, fn)
                    end

                    L = linear_assignment(C)
                    @test isfeasible(L)
                    @test compute_cost(L, C) == cost

                    @test typeof(L) === typeof(LC)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LC, fn)
                    end
                end

                L = LAWorkspaceCSC(Tv, D)
                linear_assignment!(L, D, false, false)
                @test isfeasible(L)
                @test compute_cost(L, D) == cost

                if Tv === T
                    @test typeof(L) === typeof(LD)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LD, fn)
                    end

                    L = linear_assignment(D)
                    @test isfeasible(L)
                    @test compute_cost(L, D) == cost

                    @test typeof(L) === typeof(LD)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LD, fn)
                    end
                end

                L = LAWorkspaceCSC(Tv, S)
                linear_assignment!(L, S, false, false)
                @test isfeasible(L)
                @test compute_cost(L, S) == cost

                if Tv === T
                    @test typeof(L) === typeof(LS)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LS, fn)
                    end

                    L = linear_assignment(S)
                    @test isfeasible(L)
                    @test compute_cost(L, S) == cost

                    @test typeof(L) === typeof(LS)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LS, fn)
                    end
                end

                L = LAWorkspace(Tv, U)
                linear_assignment!(L, U, false, false)
                @test isfeasible(L)
                @test compute_cost(L, U) == cost

                if Tv === T
                    @test typeof(L) === typeof(LU)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LU, fn)
                    end

                    L = linear_assignment(U)
                    @test isfeasible(L)
                    @test compute_cost(L, U) == cost

                    @test typeof(L) === typeof(LU)
                    for fn in fieldnames(typeof(L))
                        getfield(L, fn) isa Ref && continue
                        @test getfield(L, fn) == getfield(LU, fn)
                    end
                end
            end
        end
    end

    @testset "Optimum Match 0 - $Tv" for Tv in REALS
        S = spzeros(Tv, 1, 1)
        add_arc!(S, 0, 1, 0)
        C = _sparse2full(S)
        D = SMatrix{1, 1}(C)
        E = MMatrix{1, 1}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S, false)
        @test isfeasible(L)
        @test compute_cost(L, S) == zero(Tv)

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C, false)
        @test isfeasible(L)
        @test compute_cost(L, C) == zero(Tv)

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test feasible
        @test compute_cost(D, DI) == zero(Tv)

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test feasible
        @test compute_cost(E, EI) == zero(Tv)
    end

    @testset "Optimum Match 1 - $Tv" for Tv in REALS
        # A problem instance containing a node with no incident arcs.
        S = spzeros(Tv, 2, 2)
        add_arc!(S, 1, 3, 0)
        C = _sparse2full(S)
        D = SMatrix{2, 2}(C)
        E = MMatrix{2, 2}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test !isfeasible(L)

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test !isfeasible(L)

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test !feasible

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test !feasible
    end

    @testset "Optimum Match 2 - $Tv" for Tv in REALS
        S = spzeros(Tv, 2, 2)
        add_arc!(S, 0, 2, 0)
        add_arc!(S, 0, 3, 2)
        add_arc!(S, 1, 2, 3)
        add_arc!(S, 1, 3, 4)
        cost = Tv(4)
        C = _sparse2full(S)
        D = SMatrix{2, 2}(C)
        E = MMatrix{2, 2}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test isfeasible(L)
        @test compute_cost(L, S) == cost

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test isfeasible(L)
        @test compute_cost(L, C) == cost

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test feasible
        @test compute_cost(D, DI) == cost

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test feasible
        @test compute_cost(E, EI) == cost
    end

    @testset "Optimum Match 3 - $Tv" for Tv in REALS
        # out of order arcs
        S = spzeros(Tv, 4, 4)
        add_arc!(S, 0, 5, 19)
        add_arc!(S, 0, 6, 47)
        add_arc!(S, 0, 7, 0)
        add_arc!(S, 1, 4, 41)
        add_arc!(S, 2, 4, 60)
        add_arc!(S, 2, 5, 15)
        add_arc!(S, 2, 7, 60)
        add_arc!(S, 3, 4, 0)
        add_arc!(S, 1, 6, 13)
        add_arc!(S, 1, 7, 41)
        cost = Tv(0 + 13 + 15 + 0)
        C = _sparse2full(S)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test isfeasible(L)
        @test compute_cost(L, S) == cost

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test isfeasible(L)
        @test compute_cost(L, C) == cost
    end

    @testset "Optimum Match 4 - $Tv" for Tv in REALS
        S = spzeros(Tv, 4, 4)
        add_arc!(S, 0, 4, 0)
        add_arc!(S, 1, 4, 60)
        add_arc!(S, 1, 5, 15)
        add_arc!(S, 1, 7, 60)
        add_arc!(S, 2, 4, 41)
        add_arc!(S, 2, 6, 13)
        add_arc!(S, 2, 7, 41)
        add_arc!(S, 3, 5, 19)
        add_arc!(S, 3, 6, 47)
        add_arc!(S, 3, 7, 0)
        cost = Tv(0 + 13 + 15 + 0)
        C = _sparse2full(S)
        D = SMatrix{4, 4}(C)
        E = MMatrix{4, 4}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test isfeasible(L)
        @test compute_cost(L, S) == cost

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test isfeasible(L)
        @test compute_cost(L, C) == cost

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test feasible
        @test compute_cost(D, DI) == cost

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test feasible
        @test compute_cost(E, EI) == cost
    end

    @testset "Optimum Match 5 - $Tv" for Tv in REALS
        S = spzeros(Tv, 4, 4)
        add_arc!(S, 0, 4, 60)
        add_arc!(S, 0, 5, 15)
        add_arc!(S, 0, 7, 60)
        add_arc!(S, 1, 4, 0)
        add_arc!(S, 2, 4, 41)
        add_arc!(S, 2, 6, 13)
        add_arc!(S, 2, 7, 41)
        add_arc!(S, 3, 5, 19)
        add_arc!(S, 3, 6, 47)
        add_arc!(S, 3, 7, 0)
        cost = Tv(0 + 13 + 15 + 0)
        C = _sparse2full(S)
        D = SMatrix{4, 4}(C)
        E = MMatrix{4, 4}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test isfeasible(L)
        @test compute_cost(L, S) == cost

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test isfeasible(L)
        @test compute_cost(L, C) == cost

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test feasible
        @test compute_cost(D, DI) == cost

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test feasible
        @test compute_cost(E, EI) == cost
    end

    @testset "Optimum Match 6 - $Tv" for Tv in REALS
        S = spzeros(Tv, 4, 4)
        add_arc!(S, 0, 4, 41)
        add_arc!(S, 0, 6, 13)
        add_arc!(S, 0, 7, 41)
        add_arc!(S, 1, 4, 60)
        add_arc!(S, 1, 5, 15)
        add_arc!(S, 1, 7, 60)
        add_arc!(S, 2, 4, 0)
        add_arc!(S, 3, 5, 19)
        add_arc!(S, 3, 6, 47)
        add_arc!(S, 3, 7, 0)
        cost = Tv(0 + 13 + 15 + 0)
        C = _sparse2full(S)
        D = SMatrix{4, 4}(C)
        E = MMatrix{4, 4}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test isfeasible(L)
        @test compute_cost(L, S) == cost

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test isfeasible(L)
        @test compute_cost(L, C) == cost

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test feasible
        @test compute_cost(D, DI) == cost

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test feasible
        @test compute_cost(E, EI) == cost
    end

    @testset "Zero Cost - $Tv" for Tv in REALS
        S = spzeros(Tv, 4, 4)
        add_arc!(S, 0, 4, 0)
        add_arc!(S, 0, 6, 0)
        add_arc!(S, 0, 7, 0)
        add_arc!(S, 1, 4, 0)
        add_arc!(S, 1, 5, 0)
        add_arc!(S, 1, 7, 0)
        add_arc!(S, 2, 4, 0)
        add_arc!(S, 3, 5, 0)
        add_arc!(S, 3, 6, 0)
        add_arc!(S, 3, 7, 0)
        C = _sparse2full(S)
        D = SMatrix{4, 4}(C)
        E = MMatrix{4, 4}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test isfeasible(L)
        @test compute_cost(L, S) == zero(Tv)

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test isfeasible(L)
        @test compute_cost(L, C) == zero(Tv)

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test feasible
        @test compute_cost(D, DI) == zero(Tv)

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test feasible
        @test compute_cost(E, EI) == zero(Tv)
    end

    @testset "Optimum Match 7 - $Tv" for Tv in REALS
        C = Tv.(TEST_PROBLEM[:C])
        S = sparse(C)
        D = SMatrix{4, 4}(C)
        E = MMatrix{4, 4}(C)
        cost = Tv(TEST_PROBLEM[:cost])
        I = TEST_PROBLEM[:I]
        J = TEST_PROBLEM[:J]

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test isfeasible(L)
        @test L.I == I
        @test L.J == J
        @test compute_cost(L, S) == cost

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test isfeasible(L)
        @test L.I == I
        @test L.J == J
        @test compute_cost(L, C) == cost

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test feasible
        @test DI == I
        @test compute_cost(D, DI) == cost

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test feasible
        @test EI == I
        @test compute_cost(E, EI) == cost
    end

    @testset "Optimum Match 8 with struct reuse - $Tv" for Tv in REALS
        Tv <: Unsigned && continue

        C1 = Tv[
            -90 -75 -75 -80
            -35 100 -55 -65
            -125 -95 -90 -105
            -45 -110 -95 -115
        ]
        D1 = SMatrix{4, 4}(C1)
        E1 = MMatrix{4, 4}(C1)
        S1 = sparse(C1)
        cost1 = Tv(-75 - 65 - 125 - 110)
        I1 = UInt32[3, 4, 1, 2]

        C2 = Tv[
            -90 -75 -75 -80
            -35 -85 -55 -65
            -125 -95 -90 -105
            -45 -110 -95 -115
        ]
        D2 = SMatrix{4, 4}(C2)
        E2 = MMatrix{4, 4}(C2)
        S2 = sparse(C2)
        cost2 = Tv(-75 - 85 - 125 - 115)
        I2 = UInt32[3, 2, 1, 4]

        L = LAWorkspaceCSC(Tv, S1)
        linear_assignment!(L, S1, true, false)
        @test isfeasible(L)
        @test L.I == I1
        @test compute_cost(L, S1) == cost1

        linear_assignment!(L, S2, true, true)
        @test isfeasible(L)
        @test L.I == I2
        @test compute_cost(L, S2) == cost2

        L = LAWorkspace(Tv, C1)
        linear_assignment!(L, C1, true, false)
        @test isfeasible(L)
        @test L.I == I1
        @test compute_cost(L, C1) == cost1

        linear_assignment!(L, C2, true, true)
        @test isfeasible(L)
        @test L.I == I2
        @test compute_cost(L, C2) == cost2

        DI, DJ, Du, Dv, feasible = linear_assignment(D1)
        @test feasible
        @test DI == I1
        @test compute_cost(D1, DI) == cost1

        DI, DJ, Du, Dv, feasible = linear_assignment(D2)
        @test feasible
        @test DI == I2
        @test compute_cost(D2, DI) == cost2

        EI, EJ, Eu, Ev, feasible = linear_assignment(E1)
        @test feasible
        @test EI == I1
        @test compute_cost(E1, EI) == cost1

        EI, EJ, Eu, Ev, feasible = linear_assignment(E2)
        @test feasible
        @test EI == I2
        @test compute_cost(E2, EI) == cost2
    end

    @testset "Infeasible Problems - $Tv" for Tv in REALS
        # No arcs in the graph at all.
        S = spzeros(Tv, 1, 1)
        C = _sparse2full(S)
        D = SMatrix{1, 1}(C)
        E = MMatrix{1, 1}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test !isfeasible(L)

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test !isfeasible(L)

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test !feasible

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test !feasible

        # Unbalanced graph: 4 nodes on the left, 2 on the right.
        S = spzeros(Tv, 2, 4)
        add_arc!(S, 0, 2, 0)
        add_arc!(S, 1, 3, 2)
        add_arc!(S, 0, 4, 3)
        add_arc!(S, 1, 5, 4)
        C = _sparse2full(S)
        D = SMatrix{2, 4}(C)
        E = MMatrix{2, 4}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test !isfeasible(L)

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test !isfeasible(L)

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test !feasible

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test !feasible

        # Balanced graph with no perfect matching.
        S = spzeros(Tv, 3, 3)
        add_arc!(S, 0, 3, 0)
        add_arc!(S, 1, 3, 2)
        add_arc!(S, 2, 3, 3)
        add_arc!(S, 0, 4, 4)
        add_arc!(S, 0, 5, 5)
        C = _sparse2full(S)
        D = SMatrix{3, 3}(C)
        E = MMatrix{3, 3}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test !isfeasible(L)

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test !isfeasible(L)

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test !feasible

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test !feasible

        # Another balanced graph with no perfect matching, but with plenty
        # of in/out degree for each node.
        S = spzeros(Tv, 5, 5)
        add_arc!(S, 0, 5, 0)
        add_arc!(S, 0, 6, 2)
        add_arc!(S, 1, 5, 3)
        add_arc!(S, 1, 6, 4)
        add_arc!(S, 2, 5, 4)
        add_arc!(S, 2, 6, 4)
        add_arc!(S, 3, 7, 4)
        add_arc!(S, 3, 8, 4)
        add_arc!(S, 3, 9, 4)
        add_arc!(S, 4, 7, 4)
        add_arc!(S, 4, 8, 4)
        add_arc!(S, 4, 9, 4)
        C = _sparse2full(S)
        D = SMatrix{5, 5}(C)
        E = MMatrix{5, 5}(C)

        L = LAWorkspaceCSC(Tv, S)
        linear_assignment!(L, S)
        @test !isfeasible(L)

        L = LAWorkspace(Tv, C)
        linear_assignment!(L, C)
        @test !isfeasible(L)

        DI, DJ, Du, Dv, feasible = linear_assignment(D)
        @test !feasible

        EI, EJ, Eu, Ev, feasible = linear_assignment(E)
        @test !feasible
    end

    @testset "Hard Problem - $Tv" for Tv in REALS
        # The following test computes assignments on the instances described in:
        # Robert E. Machol and Michael Wien, "Errata: A Hard Assignment Problem",
        # Operations Research, vol. 25, p. 364, 1977.
        # http://www.jstor.org/stable/169842
        #
        # Such instances proved difficult for the Hungarian method.
        @testset "N = $n" for n in (10, 100, 1000)
            (n-1)*(n-1) >= typemax(Tv) && continue
            S, C = machol_wien(Tv, n, n)

            L = LAWorkspaceCSC(Tv, S)
            linear_assignment!(L, S)
            @test isfeasible(L)

            for j in 1:n
                i = Int(L.I[j])
                left = i - 1
                right = n + j - 1
                @test (left + right) == (2*n - 1)
            end

            L = LAWorkspace(Tv, C)
            linear_assignment!(L, C)
            @test isfeasible(L)

            for j in 1:n
                i = Int(L.I[j])
                left = i - 1
                right = n + j - 1
                @test (left + right) == (2*n - 1)
            end

            if n == 10
                D = SMatrix{10, 10}(C)
                E = MMatrix{10, 10}(C)

                DI, DJ, Du, Dv, feasible = linear_assignment(D)
                @test feasible

                for j in 1:n
                    i = Int(DI[j])
                    left = i - 1
                    right = n + j - 1
                    @test (left + right) == (2*n - 1)
                end

                EI, EJ, Eu, Ev, feasible = linear_assignment(E)
                @test feasible

                for j in 1:n
                    i = Int(EI[j])
                    left = i - 1
                    right = n + j - 1
                    @test (left + right) == (2*n - 1)
                end
            end
        end
    end
end
