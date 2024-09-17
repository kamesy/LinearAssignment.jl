# LinearAssignment

[![Build Status](https://github.com/kamesy/LinearAssignment.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kamesy/LinearAssignment.jl/actions/workflows/CI.yml?query=branch%3Amain)

A modified Jonker-Volgenant algorithm for solving the rectangular assignment problem [1,2].

## Installation

LinearAssignment requires Julia v1.6 or later.

```julia
julia> ]add LinearAssignment
```

## Usage

Dense problems:
```julia
using LinearAssignment

# cost matrix
C = [
    90 76 75 80
    35 85 55 65
    125 95 90 105
    45 110 95 115
]

# solve
L = linear_assignment(C)

# pre-allocate workspace
L = LAWorkspace(C)
L = LAWorkspace(Int, C)
L = LAWorkspace(Float64, size(C)...)

# solve in-place
linear_assignment!(L, C)

# check feasibility
isfeasible(L)

# compute cost
cost = compute_cost(L, C)

# remove edge
C[4,1] = typemax(eltype(C))
linear_assignment!(L, C)

# assigned row and column indices, C[I[j],j], C[i,J[i]]
I = L.I
J = L.J

# row and column dual vectors
u = L.u
v = L.v

# StaticArrays
using StaticArrays

S = SMatrix{4, 4}(C)

I, J, u, v, feasible = linear_assignment(S)
cost = compute_cost(C, I)
```

Sparse problems:
```julia
using SparseArrays
using SparseArrays: getcolptr

# cost matrix
C = sparse([
    0 76 75 80
    0 85 55 65
    125 95 0 0
    45 0 95 115
])

# solve
L = linear_assignment(C)

# pre-allocate workspace
L = LAWorkspaceCSC(C)
L = LAWorkspaceCSC(Int, C)
L = LAWorkspaceCSC(Float64, size(C)...)

# solve in-place
linear_assignment!(L, C)

# pass vectors directly
m, n = size(C)
colptr = getcolptr(C)
rowval = rowvals(C)
nzval = nonzeros(C)

linear_assignment!(L, m, n, colptr, rowval, nzval)

# check feasibility
isfeasible(L)

# compute cost
cost = compute_cost(L, C)
cost = compute_cost(L, m, n, colptr, rowval, nzval)

# assigned row and column indices, C[I[j],j], C[i,J[i]]
I = L.I
J = L.J

# row and column dual vectors
u = L.u
v = L.v
```

## References
[1] DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679-1696,
    August 2016, https://doi.org/10.1109/TAES.2016.140952

[2] R Jonker, T Volgenant. A shortest augmenting path algorithm for dense and
    sparse linear assignment problems. DGOR/NSOR. Operations Research Proceedings,
    vol 1987. Springer, Berlin, Heidelberg.
    https://doi.org/10.1007/978-3-642-73778-7_164
