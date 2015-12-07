# simple program to test the new k-means (not ready yet)

using Base.Test
using Clustering
using Compat

srand(34568)

m = 150
n = 2000000
k = 5
maxiter = 100

#m = 60
#n = 1000000
#k = 20
#maxiter = 5000

x = rand(m, n)

# non-weighted
t1 = time()
r = kmeans(x, k; maxiter=maxiter, display=:iter)
println("time: $(time()-t1)")
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n
@test r.cweights == @compat(map(Float64, r.counts))
@test_approx_eq sum(r.costs) r.totalcost

# non-weighted (float32)
t1 = time()
r = kmeans(@compat(map(Float32, x)), k; maxiter=maxiter, display=:iter)
println("time: $(time()-t1)")
@test isa(r, KmeansResult{Float32})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n
@test r.cweights == @compat(map(Float64, r.counts))
@test_approx_eq sum(r.costs) r.totalcost

# weighted
w = rand(n)
t1 = time()
r = kmeans(x, k; maxiter=maxiter, weights=w, display=:iter)
println("time: $(time()-t1)")
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n

cw = zeros(k)
for i = 1:n
	cw[r.assignments[i]] += w[i]
end
@test_approx_eq r.cweights cw

@test_approx_eq dot(r.costs, w) r.totalcost
