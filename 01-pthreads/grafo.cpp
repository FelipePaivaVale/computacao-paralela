#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <mutex>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <thread>

std::unordered_map<int, std::vector<int>> graph, reverse_graph;
std::mutex mtx;

int nodes = 0, edges = 0;
int largest_wcc_nodes = 0, largest_wcc_edges = 0;
int largest_scc_nodes = 0, largest_scc_edges = 0;
double average_clustering_coefficient = 0.0;
long long triangles = 0;
double fraction_of_closed_triangles = 0.0;
int diameter = 0;

void load_graph(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        int from, to;
        iss >> from >> to;
        {
            std::lock_guard<std::mutex> lock(mtx);
            graph[from].push_back(to);
            reverse_graph[to].push_back(from);
            if (graph.find(to) == graph.end()) {
                graph[to] = {};
            }
        }
        edges++;
    }
    nodes = graph.size();
}

void bfs_wcc(int start, std::unordered_set<int>& visited, int& component_nodes, int& component_edges) {
    std::queue<int> q;
    q.push(start);
    visited.insert(start);
    component_nodes = 0;
    component_edges = 0;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        component_nodes++;
        for (const auto& neighbor : graph[node]) {
            component_edges++;
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
}

void dfs(int node, std::unordered_set<int>& visited, std::stack<int>& finish_stack) {
    visited.insert(node);
    for (const auto& neighbor : graph[node]) {
        if (visited.find(neighbor) == visited.end()) {
            dfs(neighbor, visited, finish_stack);
        }
    }
    finish_stack.push(node);
}

void reverse_dfs(int node, std::unordered_set<int>& visited, int& component_nodes, int& component_edges) {
    visited.insert(node);
    component_nodes++;
    for (const auto& neighbor : reverse_graph[node]) {
        component_edges++;
        if (visited.find(neighbor) == visited.end()) {
            reverse_dfs(neighbor, visited, component_nodes, component_edges);
        }
    }
}

void calculate_clustering_and_triangles() {
    long long total_closed_triangles = 0;
    long long total_possible_triangles = 0;

    for (const auto& [node, neighbors] : graph) {
        std::unordered_set<int> neighbor_set(neighbors.begin(), neighbors.end());
        int local_triangles = 0;
        for (const auto& neighbor : neighbors) {
            for (const auto& second_neighbor : graph[neighbor]) {
                if (neighbor_set.find(second_neighbor) != neighbor_set.end()) {
                    local_triangles++;
                }
            }
        }
        local_triangles /= 2;
        total_closed_triangles += local_triangles;
        total_possible_triangles += neighbors.size() * (neighbors.size() - 1) / 2;
    }

    std::lock_guard<std::mutex> lock(mtx);
    triangles = total_closed_triangles;
    fraction_of_closed_triangles = total_possible_triangles > 0 ? (double)triangles / total_possible_triangles : 0.0;
    average_clustering_coefficient = fraction_of_closed_triangles;
}

int bfs_diameter(int start) {
    std::queue<int> q;
    std::unordered_map<int, int> distances;
    q.push(start);
    distances[start] = 0;
    int max_distance = 0;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        for (const auto& neighbor : graph[node]) {
            if (distances.find(neighbor) == distances.end()) {
                distances[neighbor] = distances[node] + 1;
                max_distance = std::max(max_distance, distances[neighbor]);
                q.push(neighbor);
            }
        }
    }
    return max_distance;
}

void calculate_diameter() {
    int max_diameter = 0;
    for (const auto& [node, _] : graph) {
        int local_diameter = bfs_diameter(node);
        max_diameter = std::max(max_diameter, local_diameter);
    }
    std::lock_guard<std::mutex> lock(mtx);
    diameter = max_diameter;
}

int main() {
    std::string filename = "web-Google.txt";

    load_graph(filename);

    std::unordered_set<int> visited;
    for (const auto& [node, _] : graph) {
        if (visited.find(node) == visited.end()) {
            int component_nodes = 0, component_edges = 0;
            bfs_wcc(node, visited, component_nodes, component_edges);
            if (component_nodes > largest_wcc_nodes) {
                largest_wcc_nodes = component_nodes;
                largest_wcc_edges = component_edges / 2;
            }
        }
    }

    std::thread clustering_thread(calculate_clustering_and_triangles);

    std::thread diameter_thread(calculate_diameter);

    clustering_thread.join();
    diameter_thread.join();

    std::cout << "{\n  \"graph_metrics\": {\n"
              << "    \"nodes\": " << nodes << ",\n"
              << "    \"edges\": " << edges << ",\n"
              << "    \"largest_wcc\": {\n"
              << "      \"nodes\": " << largest_wcc_nodes << ",\n"
              << "      \"fraction_of_total_nodes\": " << std::fixed << std::setprecision(3) << (double)largest_wcc_nodes / nodes << ",\n"
              << "      \"edges\": " << largest_wcc_edges << ",\n"
              << "      \"fraction_of_total_edges\": " << (double)largest_wcc_edges / edges << "\n"
              << "    },\n"
              << "    \"average_clustering_coefficient\": " << average_clustering_coefficient << ",\n"
              << "    \"triangles\": " << triangles << ",\n"
              << "    \"fraction_of_closed_triangles\": " << fraction_of_closed_triangles << ",\n"
              << "    \"diameter\": " << diameter << "\n"
              << "  }\n}\n";

    return 0;
}
