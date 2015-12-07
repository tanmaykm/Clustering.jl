# simple program to test the new k-means (not ready yet)

using Compat
using Base.Test
using Clustering
using DistributedArrays

# non-weighted
function kmeanstest(T=Float64, m=60, n=10^6, k=20, maxiter=5000, weights=nothing, display=:iter)
    @everywhere srand(34568)

    x = rand(T, m, n)

    # create k*20 clusters
    for j in 1:n
        x[:,j] += 10*(1+j%(20*k))
    end
    
    gc()
    @everywhere srand(34568)
    t1 = time()
    r = kmeans(x, k; maxiter=maxiter, weights=weights, display=display)
    tsingle = time() - t1

    x = distribute(x)
    if weights !== nothing
        weights = distribute(weights)
    end
    gc()
    @everywhere srand(34568)
    t1 = time()
    rd = kmeans(x, k; maxiter=maxiter, weights=weights, display=display)
    tdist = time() - t1

    gc()
    @everywhere srand(34568)
    t1 = time()
    l = ceil(Int, 2 * k)
    m = 3
    rp = kmeans(x, k; init=KmparAlg(l,m), maxiter=maxiter, weights=weights, display=display)
    tpar = time() - t1

    @test convert(Array, rd.assignments) == r.assignments
    @test_approx_eq convert(Array, rd.centers) r.centers
    @test_approx_eq convert(Array, rd.costs) r.costs

    dcenters_sorted = rd.centers[:,sortperm(reshape(sum(rd.centers, 1), k))]
    pcenters_sorted = rp.centers[:,sortperm(reshape(sum(rp.centers, 1), k))]
    @test_approx_eq dcenters_sorted pcenters_sorted

    if display !== :none
        println("parallel: $tpar, distributed: $tdist, singlenode: $tsingle")
    end
    nothing
end

println("\nnonweighted float64")
kmeanstest(Float64, 15, 10^4, 5, 1000, nothing, :none)
kmeanstest(Float64, 15, 10^4, 5, 1000, nothing, :final)
#kmeanstest(Float64, 60, 10^6, 50, 5000, nothing, :iter)

println("\nweighted float64")
kmeanstest(Float64, 15, 10^4, 5, 1000, rand(10^4), :final)

println("\nnonweighted float32")
kmeanstest(Float32, 15, 10^4, 5, 1000, nothing, :final)
