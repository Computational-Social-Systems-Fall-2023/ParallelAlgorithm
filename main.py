from mpi4py import MPI
import networkx as nx
import numpy as np


def load_graph(filename):
    # Load the graph from the specified file
    return nx.read_edgelist(filename)


def calculate_closeness_centrality(graph, start_node, end_node):
    # Calculate closeness centrality for nodes in the specified range
    closeness_centralities = {}
    for node in range(start_node, end_node):
        closeness_centralities[node] = nx.closeness_centrality(graph, u=node)
    return closeness_centralities


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load the graph
    graph_filename = "facebook_combined.txt"  # Change this to the appropriate filename
    graph = load_graph(graph_filename)

    # Get the number of nodes in the graph
    num_nodes = graph.number_of_nodes()

    # Determine the range of nodes to process for each processor
    nodes_per_processor = num_nodes // size
    start_node = rank * nodes_per_processor
    end_node = (rank + 1) * nodes_per_processor if rank < size - 1 else num_nodes

    # Calculate closeness centrality for the assigned nodes
    local_closeness_centralities = calculate_closeness_centrality(graph, start_node, end_node)

    # Gather results from all processors to processor 0
    all_closeness_centralities = comm.gather(local_closeness_centralities, root=0)

    if rank == 0:
        # Combine results from all processors
        combined_closeness_centralities = {}
        for centrality_dict in all_closeness_centralities:
            combined_closeness_centralities.update(centrality_dict)

        # Output results for processor 0
        with open("output.txt", "w") as output_file:
            for node, centrality in combined_closeness_centralities.items():
                output_file.write(f"Node {node}: Closeness Centrality = {centrality}\n")

            # Print top 5 nodes with highest centrality
            top_nodes = sorted(combined_closeness_centralities, key=combined_closeness_centralities.get, reverse=True)[
                        :5]
            output_file.write("\nTop 5 Nodes with Highest Closeness Centrality:\n")
            for node in top_nodes:
                output_file.write(f"Node {node}: Closeness Centrality = {combined_closeness_centralities[node]}\n")

            # Print average centrality
            avg_centrality = np.mean(list(combined_closeness_centralities.values()))
            output_file.write(f"\nAverage Closeness Centrality: {avg_centrality}\n")


if __name__ == "__main__":
    main()
