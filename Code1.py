import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

# ---------- 1. Graph constructors ----------
def make_line(n_nodes):
    """Creates a simple line (path) graph."""
    return nx.path_graph(n_nodes)

def make_ring(n_nodes):
    """Creates a closed ring graph."""
    G = nx.Graph()
    for i in range(1, n_nodes):
        G.add_edge(i - 1, i)
    G.add_edge(n_nodes - 1, 0)
    return G

def make_star(n_leaves):
    """Creates a star graph with one central node."""
    return nx.star_graph(n_leaves)

def make_lollipop(n_up=3):
    """Creates a 'lollipop' graph: 3 horizontal nodes and n_up vertical nodes above the center."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])  # horizontal base
    for i in range(3, 3 + n_up):
        if i == 3:
            G.add_edge(1, i)             # connect first vertical
        else:
            G.add_edge(i - 1, i)         # extend upward
    return G

def make_random_graph(n_nodes=10, connection_prob=0.3, seed=42):
    """Creates a random undirected connected graph (Erdős–Rényi model)."""
    G = nx.erdos_renyi_graph(n=n_nodes, p=connection_prob, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.erdos_renyi_graph(n=n_nodes, p=connection_prob, seed=seed)
    return G


# ---------- 2. Scattering operator ----------
def scattering_operator(G):
    """Builds the unitary scattering (step) operator U for a given graph with reflection at degree-1 nodes."""
    edges = list(G.edges())
    directed = [(u, v) for (u, v) in edges] + [(v, u) for (u, v) in edges]
    index = {e: i for i, e in enumerate(directed)}
    U = np.zeros((len(directed), len(directed)), dtype=complex)

    for (v, w) in directed:
        d = len(list(G.neighbors(w)))
        if d == 1:
            # Reflective boundary (for unitarity)
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
def evolve(U, directed, start_vertex, steps, marked_vertex):
    """Evolves the quantum walk and returns the probability of the marked vertex."""
    psi = np.zeros(len(directed), dtype=complex)
    out_edges = [i for i, (v, w) in enumerate(directed) if v == start_vertex]
    psi[out_edges] = 1 / np.sqrt(len(out_edges))
    probs_marked = []

    for _ in range(steps):
        psi = U @ psi
        marked_edges = [i for i, (v, w) in enumerate(directed)
                        if v == marked_vertex or w == marked_vertex]
        Pm = np.sum(np.abs(psi[marked_edges]) ** 2)
        probs_marked.append(Pm)
    return np.array(probs_marked)


def vertex_probabilities(U, directed, G, start_vertex, steps):
    """Computes probability distribution over all vertices at each time step."""
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
def classical_random_walk(G, start_vertex=0, steps=120, marked_vertex=None):
    """Simulates a classical random walk using transition probabilities."""
    vertices = list(G.nodes())
    n = len(vertices)
    adj = nx.to_numpy_array(G)
    deg = adj.sum(axis=1)
    transition_matrix = (adj.T / deg).T

    p = np.zeros(n)
    p[start_vertex] = 1.0

    probs_marked = []
    for _ in range(steps):
        p = transition_matrix @ p
        if marked_vertex is not None:
            probs_marked.append(p[marked_vertex])
        else:
            probs_marked.append(p.copy())
    return np.array(probs_marked)


# ---------- 5. Simulation and saving ----------
def simulate_and_save(G, name, start_vertex=0, steps=120):
    """Runs both quantum and classical walks and saves results."""
    os.makedirs("results", exist_ok=True)
    marked = max(G.nodes())

    # Quantum simulation
    U, directed = scattering_operator(G)
    quantum_probs = evolve(U, directed, start_vertex, steps, marked)
    prob_matrix, _ = vertex_probabilities(U, directed, G, start_vertex, 80)

    # Save quantum CSV
    df_quantum = pd.DataFrame({"time": np.arange(len(quantum_probs)), "P_marked_quantum": quantum_probs})
    df_quantum.to_csv(f"results/{name}_quantum_probability.csv", index=False)

    # Plot quantum results
    plt.figure()
    plt.plot(df_quantum["time"], df_quantum["P_marked_quantum"], color="blue")
    plt.xlabel("Time step")
    plt.ylabel(f"P(marked={marked})")
    plt.title(f"Quantum Walk Probability ({name})")
    plt.grid(True)
    plt.savefig(f"results/{name}_quantum_prob_vs_time.png", dpi=300)
    plt.close()

    # Heat map
    plt.imshow(prob_matrix, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Probability")
    plt.xlabel("Time step")
    plt.ylabel("Vertex index")
    plt.title(f"Quantum Walk Heat Map ({name})")
    plt.savefig(f"results/{name}_quantum_heatmap.png", dpi=300)
    plt.close()

    # Classical simulation
    classical_probs = classical_random_walk(G, start_vertex, steps, marked)
    df_classical = pd.DataFrame({"time": np.arange(len(classical_probs)), "P_marked_classical": classical_probs})
    df_classical.to_csv(f"results/{name}_classical_probability.csv", index=False)

    # Quantum vs Classical Comparison
    plt.figure()
    plt.plot(df_quantum["time"], df_quantum["P_marked_quantum"], label="Quantum Walk", color="blue")
    plt.plot(df_classical["time"], df_classical["P_marked_classical"], label="Classical Walk", linestyle="--", color="orange")
    plt.xlabel("Time step")
    plt.ylabel(f"P(marked={marked})")
    plt.title(f"Quantum vs Classical Walk Comparison ({name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{name}_quantum_vs_classical.png", dpi=300)
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
    marked_vertex = max(G.nodes())

    # Save colored graph visualization
    plt.figure()
    pos = nx.spring_layout(G, seed=42)
    node_colors = [
        "limegreen" if node == start_vertex else
        "red" if node == marked_vertex else
        "skyblue"
        for node in G.nodes()
    ]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray",
            node_size=800, font_size=10, font_weight="bold")
    plt.title(f"{name} Graph (Green=start, Red=marked)")
    plt.savefig(f"results/{name}_graph_structure.png", dpi=300)
    plt.close()

    simulate_and_save(G, name, start_vertex=start_vertex)
