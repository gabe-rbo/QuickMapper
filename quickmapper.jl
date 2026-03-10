using Random
Random.seed!(24)

function quick_mapper_jl(G_raw::PyDict{Any, Any}, max_loops::Int=1, min_modularity_gain::Float64=1e-6)
    # Extract vertices and edges from the Python dictionary, casting to native 
    # Julia integers to ensure type stability and high performance.
    V_py = G_raw["V"]
    E_py = G_raw["E"]
    V = [Int(v) for v in V_py]
    E = [(Int(e[1]), Int(e[2])) for e in E_py]

    # Build an adjacency list representation of the graph.
    # This allows fast lookups of a node's neighbors during the clustering phase.
    adj = Dict{Int, Vector{Int}}(v => Int[] for v in V)
    for (u, v) in E
        push!(adj[u], v)
        push!(adj[v], u)
    end

    m = length(E)

    # Initialize labels: every vertex starts in its own individual cluster (community).
    L = Dict{Int, Int}(v => v for v in V)
    
    # Handle the edge case of a completely disconnected graph to avoid division by zero.
    if m == 0
        return Dict("V" => collect(Set(values(L))), "E" => []), L
    end

    # Precompute the degree (number of connections) for each vertex. 
    # This is heavily used in the modularity calculation.
    degree = Dict{Int, Int}(v => length(adj[v]) for v in V)

    num_of_loops = 0
    modularity_gain = 1000.0
    
    # Preallocate arrays and constants outside the loop to save memory and time.
    best_labels = Int[]
    two_m = 2.0 * m

    # Label Propagation Loop: Iteratively group nodes into communities.
    while modularity_gain > min_modularity_gain && num_of_loops < max_loops
        # Randomizing the vertex order prevents infinite loops and bias.
        vertex_order = collect(V)
        shuffle!(vertex_order)

        for vertex in vertex_order
            neighbors = adj[vertex]
            if isempty(neighbors)
                continue
            end

            # Identify all unique communities (labels) present among this node's neighbors.
            # We also include the node's current label as an option to stay put.
            NbrLabelSet_vertex = Set{Int}()
            push!(NbrLabelSet_vertex, L[vertex])
            for nbr in neighbors
                push!(NbrLabelSet_vertex, L[nbr])
            end

            empty!(best_labels)
            max_contribution = -Inf
            deg_v = degree[vertex]

            # Evaluate moving 'vertex' into each neighboring community.
            # The 'contribution' measures how much better the network's modularity
            # becomes if we assign 'vertex' to 'label'.
            for label in NbrLabelSet_vertex
                contribution = 0.0
                for j in neighbors
                    if L[j] == label
                        # Modularity formula component: Actual edge minus expected edges
                        contribution += (1.0 - (deg_v * degree[j]) / two_m)
                    end
                end

                # Keep track of the label(s) that yield the highest modularity gain.
                if contribution > max_contribution
                    max_contribution = contribution
                    empty!(best_labels)
                    push!(best_labels, label)
                elseif contribution == max_contribution
                    push!(best_labels, label)
                end
            end

            # Assign the vertex to the best community found. Break ties randomly.
            L[vertex] = rand(best_labels)
        end

        # In this simplified demonstration, we force an exit after max_loops.
        modularity_gain = 0.0 
        num_of_loops += 1
    end

    # Build the simplified "meta-graph".
    # Vertices that share the same label are merged into a single node.
    E_simple = Set{Tuple{Int, Int}}()
    for vertex in V
        for nbr in adj[vertex]
            lv = L[vertex]
            lnbr = L[nbr]
            
            # If a vertex and its neighbor ended up in different communities,
            # we create an edge between those communities in the new simplified graph.
            if lv != lnbr
                # minmax orders the tuple (smaller, larger) so (A, B) and (B, A) 
                # are treated as the exact same undirected edge.
                push!(E_simple, minmax(lv, lnbr))
            end
        end
    end

    # Package the simplified graph and the mapping of original nodes to new clusters.
    G_simple = Dict("V" => collect(Set(values(L))), "E" => collect(E_simple))

    return G_simple, L
