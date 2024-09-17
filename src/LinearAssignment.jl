module LinearAssignment


using Base: RefValue, require_one_based_indexing
using Base.Checked: sub_with_overflow

using SparseArrays: SparseMatrixCSC
using SparseArrays: getcolptr, nonzeros, rowvals
using StaticArrays: MMatrix, SMatrix, SVector, @MVector


export LAWorkspace, LAWorkspaceCSC
export linear_assignment, linear_assignment!
export compute_cost, isfeasible

include("dense.jl")
include("sparse.jl")


end # module
