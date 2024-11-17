#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <iomanip>
#include <cmath>
#include <algorithm>

std::unordered_map<int, std::vector<int>> graph;
int world_rank, world_size;

// Função para carregar o grafo no processo mestre
void load_graph(const std::string& filename) {
    if (world_rank == 0) {
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            if (line[0] == '#') continue;
            std::istringstream iss(line);
            int from, to;
            iss >> from >> to;
            graph[from].push_back(to);
            if (graph.find(to) == graph.end()) {
                graph[to] = {};
            }
        }
    }
}

// Distribuir o grafo entre os processos
void distribute_graph() {
    if (world_rank == 0) {
        for (int dest = 1; dest < world_size; dest++) {
            for (const auto& [node, neighbors] : graph) {
                int node_copy = node;  // Criação de uma cópia sem const
                MPI_Send(&node_copy, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

                int size = neighbors.size();
                MPI_Send(&size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

                // Criação de cópia não constante do vetor de vizinhos
                std::vector<int> neighbors_copy = neighbors;
                MPI_Send(neighbors_copy.data(), size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
            int end_signal = -1;
            MPI_Send(&end_signal, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        int node;
        while (true) {
            MPI_Recv(&node, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (node == -1) break;
            int size;
            MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> neighbors(size);
            MPI_Recv(neighbors.data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            graph[node] = neighbors;
        }
    }
}

// BFS para calcular o maior componente conexo (WCC e SCC)
std::pair<int, int> bfs_component(int start, std::unordered_set<int>& visited, bool reverse) {
    std::queue<int> q;
    q.push(start);
    visited.insert(start);

    int component_nodes = 0;
    int component_edges = 0;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        component_nodes++;

        const auto& neighbors = reverse ? graph[node] : graph[node];
        for (const auto& neighbor : neighbors) {
            component_edges++;
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    return {component_nodes, component_edges};
}

// Calcula o coeficiente médio de clustering
double calculate_clustering_coefficient() {
    int local_closed_triplets = 0, local_total_triplets = 0;

    for (const auto& [node, neighbors] : graph) {
        int size = neighbors.size();
        if (size < 2) continue;

        local_total_triplets += size * (size - 1) / 2;
        for (size_t i = 0; i < neighbors.size(); i++) {
            for (size_t j = i + 1; j < neighbors.size(); j++) {
                if (std::find(graph[neighbors[i]].begin(), graph[neighbors[i]].end(), neighbors[j]) != graph[neighbors[i]].end()) {
                    local_closed_triplets++;
                }
            }
        }
    }

    int global_closed_triplets = 0, global_total_triplets = 0;
    MPI_Reduce(&local_closed_triplets, &global_closed_triplets, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_triplets, &global_total_triplets, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    return (global_total_triplets > 0) ? (double)global_closed_triplets / global_total_triplets : 0.0;
}

// Função principal
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string filename = "web-Google.txt";

    // Carregar e distribuir o grafo
    load_graph(filename);
    distribute_graph();

    // Calcular maior WCC
    std::unordered_set<int> visited;
    int local_wcc_nodes = 0, local_wcc_edges = 0;

    for (const auto& [node, _] : graph) {
        if (visited.find(node) == visited.end()) {
            auto [component_nodes, component_edges] = bfs_component(node, visited, false);
            if (component_nodes > local_wcc_nodes) {
                local_wcc_nodes = component_nodes;
                local_wcc_edges = component_edges;
            }
        }
    }

    int global_wcc_nodes, global_wcc_edges;
    MPI_Reduce(&local_wcc_nodes, &global_wcc_nodes, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_wcc_edges, &global_wcc_edges, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Calcular maior SCC
    visited.clear();
    int local_scc_nodes = 0, local_scc_edges = 0;

    for (const auto& [node, _] : graph) {
        if (visited.find(node) == visited.end()) {
            auto [component_nodes, component_edges] = bfs_component(node, visited, true);
            if (component_nodes > local_scc_nodes) {
                local_scc_nodes = component_nodes;
                local_scc_edges = component_edges;
            }
        }
    }

    int global_scc_nodes, global_scc_edges;
    MPI_Reduce(&local_scc_nodes, &global_scc_nodes, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_scc_edges, &global_scc_edges, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Coeficiente médio de clustering
    double avg_clustering_coefficient = calculate_clustering_coefficient();

    // Processo mestre exibe os resultados
    if (world_rank == 0) {
        std::cout << "{\n  \"graph_metrics\": {\n"
                  << "    \"nodes\": " << graph.size() << ",\n"
                  << "    \"edges\": " << global_wcc_edges << ",\n"
                  << "    \"largest_wcc\": {\n"
                  << "      \"nodes\": " << global_wcc_nodes << ",\n"
                  << "      \"fraction_of_total_nodes\": " << (double)global_wcc_nodes / graph.size() << ",\n"
                  << "      \"edges\": " << global_wcc_edges << ",\n"
                  << "      \"fraction_of_total_edges\": " << (double)global_wcc_edges / global_wcc_edges << "\n"
                  << "    },\n"
                  << "    \"largest_scc\": {\n"
                  << "      \"nodes\": " << global_scc_nodes << ",\n"
                  << "      \"fraction_of_total_nodes\": " << (double)global_scc_nodes / graph.size() << ",\n"
                  << "      \"edges\": " << global_scc_edges << ",\n"
                  << "      \"fraction_of_total_edges\": " << (double)global_scc_edges / global_wcc_edges << "\n"
                  << "    },\n"
                  << "    \"average_clustering_coefficient\": " << avg_clustering_coefficient << "\n"
                  << "  }\n}\n";
    }

    MPI_Finalize();
    return 0;
}