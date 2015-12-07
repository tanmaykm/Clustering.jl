using DataFrames
using Clustering
using DistributedArrays

function prepare(datapath)
    inp = joinpath(datapath, "kddcup.data")
    out = joinpath(datapath, "kddtrain.csv")

    info("reading data from $inp")
    d = readtable(inp)

    names!(d, [:duration, :protocol_type, :service, :flag, :src_bytes, :dst_bytes, :land, :wrong_fragment, :urgent, :hot, :num_failed_logins, :logged_in, :num_compromised, :root_shell, :su_attempted, :num_root, :num_file_creations, :num_shells, :num_access_files, :num_outbound_cmds, :is_host_login, :is_guest_login, :count, :srv_count, :serror_rate, :srv_serror_rate, :rerror_rate, :srv_rerror_rate, :same_srv_rate, :diff_srv_rate, :srv_diff_host_rate, :dst_host_count, :dst_host_srv_count, :dst_host_same_srv_rate, :dst_host_diff_srv_rate, :dst_host_same_src_port_rate, :dst_host_srv_diff_host_rate, :dst_host_serror_rate, :dst_host_srv_serror_rate, :dst_host_rerror_rate, :dst_host_srv_rerror_rate, :typ])

    info("pooling category columns...")
    category_columns = [:protocol_type, :service, :flag, :typ]
    pool!(d, category_columns)

    for c in category_columns
        println("levels of $c")
        println(levels(d[c]))
    end

    info("creating new columns...")
    for lev in levels(d[:protocol_type])
        newcol = symbol("protocol_$lev")
        d[newcol] = convert(Array{Int}, d[:protocol_type] .== lev)
    end

    info("deleting  unnecessary columns...")
    delete!(d, [:protocol_type, :service, :flag, :typ])

    info("writing training data...")
    writetable(out, d, separator=',', header=false)
end

function findclusters(datapath, k=150, maxiter=1000, display=:iter)
    trainfile = joinpath(datapath, "kddtrain.csv")

    info("reading csv...")
    d = readcsv(trainfile)
    info("transposing...")
    d = d.'
    gc()
   
    @everywhere srand(34568)
    t1 = time()
    r = kmeans(d, k; maxiter=maxiter, display=display)
    tsingle = time() - t1
    
    info("distributing...")
    d = DistributedArrays.distribute(d)
    gc()
    @everywhere srand(34568)
    t1 = time()
    rd = kmeans(d, k; maxiter=maxiter, display=display)
    tdist = time() - t1

    gc()
    @everywhere srand(34568)
    t1 = time()
    rp = kmeans(d, k; init=KmppAlg(), maxiter=maxiter, display=display)
    tpp = time() - t1
    
    gc()
    @everywhere srand(34568)
    t1 = time()
    l = ceil(Int, 4 * k)
    m = 10
    rp = kmeans(d, k; init=KmparAlg(l,m), maxiter=maxiter, display=display)
    tpar = time() - t1
    
    #@test convert(Array, rd.assignments) == r.assignments
    #@test_approx_eq convert(Array, rd.centers) r.centers
    #@test_approx_eq convert(Array, rd.costs) r.costs

    #dcenters_sorted = rd.centers[:,sortperm(reshape(sum(rd.centers, 1), k))]
    #pcenters_sorted = rp.centers[:,sortperm(reshape(sum(rp.centers, 1), k))]
    #@test_approx_eq dcenters_sorted pcenters_sorted

    if display !== :none
        println("kmpar: $tpar, kmpp: $tpp, distributed: $tdist, singlenode: $tsingle")
    end
    nothing
end

const datapath = "/home/tan/Work/datasets/kddcup/network_intrusion_1999"
prepare(datapath)
findclusters(datapath)

