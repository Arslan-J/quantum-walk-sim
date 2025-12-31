import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

# ---------- 1. Graph constructors ----------
def make_line(n_nodes):
    return nx.path_graph(n_nodes)

def make_ring(n_nodes):
    G = nx.Graph()
    for i in range(1, n_nodes):
        G.add_edge(i - 1, i)
    G.add_edge(n_nodes - 1, 0)
    return G

def make_star(n_leaves):
    return nx.star_graph(n_leaves)

def make_lollipop(n_up=3):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    for i in range(3, 3 + n_up):
        if i == 3:
            G.add_edge(1, i)
        else:
            G.add_edge(i - 1, i)
    return G

def make_random_graph(n_nodes=10, connection_prob=0.3, seed=42):
    G = nx.erdos_renyi_graph(n=n_nodes, p=connection_prob, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.erdos_renyi_graph(n=n_nodes, p=connection_prob, seed=seed)
    return G


# ---------- 2. Scattering operator ----------
def scattering_operator(G):
    edges = list(G.edges())
    directed = [(u, v) for (u, v) in edges] + [(v, u) for (u, v) in edges]
    index = {e: i for i, e in enumerate(directed)}
    U = np.zeros((len(directed), len(directed)), dtype=complex)

    for (v, w) in directed:
        d = len(list(G.neighbors(w)))
        if d == 1:
            U[index[(w, v)], index[(v, w)]] = -1.0
        else:
            r = (d - 2) / d
            t = 2 / d
            for x in G.neighbors(w):
                if x == v:
                    U[index[(w, x)], index[(v, w)]] = -r
                else:
                    U[index[(w, x)], index[(v, w)]] = t
    return U, directed


# ---------- 3. Quantum time evolution ----------
def evolve(U, directed, start_vertex, steps, target_vertex):
    psi = np.zeros(len(directed), dtype=complex)

    out_edges = [i for i, (v, w) in enumerate(directed) if v == start_vertex]
    psi[out_edges] = 1 / np.sqrt(len(out_edges))

    probs_marked = []

    for _ in range(steps):
        psi = U @ psi

        incoming_edges = [
            i for i, (v, w) in enumerate(directed)
            if w == target_vertex
        ]

        Pm = np.sum(np.abs(psi[incoming_edges]) ** 2)
        probs_marked.append(Pm)

    return np.array(probs_marked)


def vertex_probabilities(U, directed, G, start_vertex, steps):
    psi = np.zeros(len(directed), dtype=complex)
    out_edges = [i for i, (v, w) in enumerate(directed) if v == start_vertex]
    psi[out_edges] = 1 / np.sqrt(len(out_edges))
    vertices = list(G.nodes())
    v_index = {v: i for i, v in enumerate(vertices)}
    prob_matrix = np.zeros((len(vertices), steps))

    for t in range(steps):
        psi = U @ psi
        for v in vertices:
            idx = v_index[v]
            v_edges = [i for i, (a, b) in enumerate(directed)
                       if a == v or b == v]
            prob_matrix[idx, t] = np.sum(np.abs(psi[v_edges]) ** 2)
    return prob_matrix, vertices


# ---------- 4. Classical random walk ----------
def classical_random_walk(G, start_vertex=0, steps=120, target_vertex=None):
    vertices = list(G.nodes())
    idx = {v: i for i, v in enumerate(vertices)}

    adj = nx.to_numpy_array(G, nodelist=vertices)
    deg = adj.sum(axis=1)
    if np.any(deg == 0):
        raise ValueError("Graph has isolated nodes; classical transition undefined.")

    transition_matrix = adj / deg[:, np.newaxis]

    p = np.zeros(len(vertices))
    p[idx[start_vertex]] = 1.0

    probs_marked = np.zeros(steps)
    prob_matrix = np.zeros((len(vertices), steps))

    for t in range(steps):
        p = p @ transition_matrix
        if target_vertex is not None:
            probs_marked[t] = p[idx[target_vertex]]
        prob_matrix[:, t] = p

    return probs_marked, prob_matrix, vertices


# ---------- 5. Simulation and saving ----------
def simulate_and_save(G, name, start_vertex=0, steps=120):
    os.makedirs("results", exist_ok=True)
    marked = max(G.nodes())

    U, directed = scattering_operator(G)
    quantum_probs = evolve(U, directed, start_vertex, steps, marked)
    quantum_probs = np.concatenate(([1.0 if start_vertex == marked else 0.0], quantum_probs))
    quantum_matrix, vertices = vertex_probabilities(U, directed, G, start_vertex, 80)

    # --- Classical ---
    classical_probs, classical_matrix, vertices = classical_random_walk(G, start_vertex, steps, marked)


    classical_probs = np.concatenate(([1.0 if start_vertex == marked else 0.0], classical_probs))
    p0 = np.zeros(len(vertices))
    p0[start_vertex] = 1.0
    classical_matrix = np.column_stack([p0, classical_matrix])
    
    # --- Combined CSV (Quantum + Classical) ---
    '''
    
    print("steps =", steps)
    print("len(quantum_probs)  =", len(quantum_probs))
    print("len(classical_probs)=", len(classical_probs))
    '''
    
    combined_df = pd.DataFrame({
        "time": np.arange(steps + 1),
        "P_marked_quantum": quantum_probs,
        "P_marked_classical": classical_probs
    })
    combined_df.to_csv(f"results/{name}_probabilities_combined.csv", index=False)

    df_full = pd.DataFrame(classical_matrix, index=vertices)
    df_full.to_csv(f"results/{name}_classical_full_vertex_probabilities.csv")

    # Comparison plot
    plt.figure(figsize=(10,6))
    plt.plot(combined_df["time"], combined_df["P_marked_quantum"], label="Quantum Walk", color="blue", linewidth=3)
    plt.plot(combined_df["time"], combined_df["P_marked_classical"], label="Classical Walk", linestyle="--", color="orange", linewidth=3)
    plt.xlabel("Time step", fontsize=16)
    plt.ylabel(f"P(marked={marked})", fontsize=16)
    plt.title(f"Quantum vs Classical Walk Comparison ({name})", fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f"results/{name}_quantum_vs_classical.pdf", bbox_inches='tight')
    plt.close()

    print(f"Finished: {name}")


# ---------- 6. Run all graph types ----------
graphs = {
    "Line": make_line(8),
    "Ring": make_ring(8),
    "Star": make_star(6),
    "Lollipop": make_lollipop(4),
    "Random": make_random_graph(10, 0.3)
}

os.makedirs("results", exist_ok=True)

for name, G in graphs.items():
    start_vertex = 0
    target_vertex = max(G.nodes())

    plt.figure()
    pos = nx.spring_layout(G, seed=42)
    node_colors = [
        "limegreen" if node == start_vertex else
        "red" if node == target_vertex else
        "skyblue"
        for node in G.nodes()
    ]
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            edge_color="gray", node_size=800, font_size=18, font_weight="bold")
    plt.title(f"{name} Graph (Green=start, Red=marked)")
    plt.savefig(f"results/{name}_graph_structure.png", dpi=300)
    plt.close()

    simulate_and_save(G, name, start_vertex=start_vertex)