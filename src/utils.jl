# Common utilities

##### common types

abstract ClusteringResult

# generic functions

nclusters(R::ClusteringResult) = length(R.counts)
counts(R::ClusteringResult) = R.counts
assignments(R::ClusteringResult) = R.assignments


##### convert weight options

conv_weights{T}(::Type{T}, n::Int, w::@compat(Void)) = nothing

function conv_weights{T}(::Type{T}, n::Int, w::Vector)
    length(w) == n || throw(DimensionMismatch("Incorrect length of weights."))
    convert(Vector{T}, w)::Vector{T}
end

function conv_weights{T}(::Type{T}, n::Int, w::DVector)
    length(w) == n || throw(DimensionMismatch("Incorrect length of weights."))
    if isa(w, DVector{T})
        return w
    end
    DArray(I->convert(Vector{T}, localpart(w)), size(w), procs(w))
end

##### convert display symbol to disp level

display_level(s::Symbol) = 
    s == :none ? 0 :
    s == :final ? 1 :
    s == :iter ? 2 :
    error("Invalid value for the option 'display'.")


##### update minimum value

function updatemin!(r::AbstractArray, x::AbstractArray)
    n = length(r)
    length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = x[i]
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end

function updatemin!(r::AbstractArray, x::AbstractMatrix)
    n = length(r)
    k = size(x, 1)
    size(x,2) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = minimum(view(x, 1:k, i))
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end

function updatemin!(r::DArray, x::DMatrix)
    DistributedArrays.map_localparts!(localr->updatemin!(localr, localpart(x)), r)
end
