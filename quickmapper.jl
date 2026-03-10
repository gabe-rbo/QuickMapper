using Random

function quick_mapper(G_raw::Dict, max_loops::Int=1, min_modularity_gain::Float64=1e-6)
    V = G_raw["V"]
    E = G_raw["E"]
    
    # Initialize adjacency list
    adj = Dict(v => Set() for v in V)
    for edge in E
        u, v = edge[1], edge[2]
        push!(adj[u], v)
        push!(adj[v], u)
    end
    
    m = length(E)
    
    L = Dict(v => v for v in V)
    
    # Edge case for an empty graph
    if m == 0
        return Dict("V" => collect(Set(values(L))), "E" => []), L
    end
    
    degree = Dict(v => length(adj[v]) for v in V)
    
    function get_P(i, j)
        return (degree[i] * degree[j]) / (2.0 * m)
    end
    
    num_of_loops = 0
    modularity_gain = 1000.0
    
    while modularity_gain > min_modularity_gain && num_of_loops < max_loops
        vertex_order = collect(V)
        shuffle!(vertex_order)
        
        for vertex in vertex_order
            neighbors = adj[vertex]
            if isempty(neighbors)
                continue
            end
            
            NbrLabelSet_vertex = Set([L[vertex]])
            for nbr in neighbors
                push!(NbrLabelSet_vertex, L[nbr])
            end
            
            best_labels = []
            max_contribution = -Inf
            
            for label in NbrLabelSet_vertex
                # Simplified local modularity gain calculation
                # Generator expressions in Julia are similar to Python
                contribution = sum((1.0 - get_P(vertex, j)) for j in neighbors if L[j] == label; init=0.0)
                
                if contribution > max_contribution
                    max_contribution = contribution
                    best_labels = [label]
                elseif contribution == max_contribution
                    push!(best_labels, label)
                end
            end
            
            L[vertex] = rand(best_labels)
        end
        
        modularity_gain = 0.0 # Force exit after set passes for this simple demonstration
        num_of_loops += 1
    end
    
    E_simple = Set{Tuple}()
    for vertex in V
        for nbr in adj[vertex]
            if L[vertex] != L[nbr]
                push!(E_simple, Tuple(sort([L[vertex], L[nbr]])))
            end
        end
    end
    
    G_simple = Dict("V" => collect(Set(values(L))), "E" => collect(E_simple))
    
    # Return both the simplified graph AND the dictionary mapping original points to clusters
    return G_simple, L 
end
