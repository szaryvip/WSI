import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.bipartite.generators import random_graph
import random
from math import floor
from time import time
from copy import deepcopy
import statistics


def generate_graph(size: int, type: str):
    """
    Generates graph depending on the desired type and returns it"""
    if type == "random":
        return nx.gnp_random_graph(size, 0.4)
    elif type == "groups":
        G = nx.gnp_random_graph(size, 0.9)
        for edge in G.edges():
            if floor(edge[0]/(size//4)) != floor(edge[1]/(size//4)):
                if random.random() < 0.95:
                    G.remove_edge(edge[0], edge[1])
        return G
    elif type == "bipartite":
        return random_graph(size//2, size//2, 0.4)


def color_graph(population, score, graph, type):
    """
    Colors best individual nodes and edges, draws graph,
    returns size of best clique"""
    score_sorted = sorted(score)
    index = score.index(score_sorted[-1])
    best_individual = population[index]
    clique = maximum_clique(best_individual, graph)[1]
    color_map = []
    for node in range(len(best_individual)):
        if node in clique:
            color_map.append("orange")
        else:
            color_map.append("blue")
    if type == "groups":
        size = len(best_individual)
        fixedpos = {0: (0, 0), (size//4): (1, 1),
                    (size//4)*2: (1, 0), (size//4)*3: (0, 1)}
        pos = nx.spring_layout(graph, fixed=fixedpos.keys(), pos=fixedpos)
        nx.draw(graph, node_color=color_map, pos=pos, with_labels=True)
        plt.show()
        return len(clique)
    nx.draw_circular(graph, node_color=color_map, with_labels=True)
    plt.show()
    return len(clique)


def init_population(g_size, p_size):
    """
    Generates random population and returns it"""
    population = []
    for i in range(p_size):
        population.append([random.randint(0, 1) for j in range(g_size)])
    return population


def maximum_clique(gens, graph):
    """
    Calculates clique and returns it with the size of it"""
    edges = [e for e in graph.edges]
    clique = []
    for index, gen in enumerate(gens):
        if gen:
            clique.append(index)
            break
    try:
        if clique[0] == len(gens)-1:
            return len(clique), clique
    except IndexError:
        return 0, clique
    for index2, gen2 in enumerate(gens[clique[0]+1:]):
        if gen2:
            in_clique = True
            for node in clique:
                if (node, index2+clique[0]+1) not in edges:
                    in_clique = False
            if in_clique:
                clique.append(index2+clique[0]+1)
    return len(clique), clique


def fitness_score(population, graph):
    """
    Assigns score for each member in population"""
    return [maximum_clique(el, graph)[0] for el in population]


def selection(population, score):
    """
    Tournament selection. Randomly choose two members and passes better one"""
    new_population = []
    i = 1
    while i <= len(population):
        i += 1
        possible = list(range(0, len(population)))
        f_num = random.choice(possible)
        possible.remove(f_num)
        s_num = random.choice(possible)
        if score[f_num] >= score[s_num]:
            new_population.append(deepcopy(population[f_num]))
        else:
            new_population.append(deepcopy(population[s_num]))
    return new_population


def mutation(population, prob_mut: int, prob_gen: int):
    """
    Arguments:
    population - population to mutate
    prob_mut - probability of individual mutation in range 0-100
    prob_gen - probability of single gen mutation in range 0-100
    Returns mutated population
    """
    new_population = []
    for ind in population:
        if random.random() <= prob_mut:
            new_gens = []
            for gen in ind:
                if random.random() <= prob_gen:
                    new_gens.append(0 if gen else 1)
                else:
                    new_gens.append(1 if gen else 0)
            new_population.append(new_gens)
        else:
            new_population.append(ind)
    return new_population


def facebook(graph_size, g_type, p_size, iterations, prob_mut, prob_gen):
    start_time = time()
    graph = generate_graph(graph_size, g_type)
    population = init_population(graph_size, p_size)
    i = 0
    score = fitness_score(population, graph)
    while i < iterations:
        select_pop = selection(population, score)
        mutated_pop = mutation(select_pop, prob_mut, prob_gen)
        score = fitness_score(mutated_pop, graph)
        population = mutated_pop
        i += 1
    best = color_graph(population, score, graph, g_type)
    func_time = time() - start_time
    return best, func_time


def main():
    # values = []
    # times = []
    # for i in range(25):
    #     value, time = facebook(50, "random", 35, 80, 0.9, 0.2)
    #     values.append(value)
    #     times.append(time)
    # print("Minimum "+str(min(values)))
    # print("Maximum "+str(max(values)))
    # print("Mean "+str(statistics.mean(values)))
    # print("Standard deviation "+str(statistics.pstdev(values)))
    # print("AVG time "+str(statistics.mean(times)))
    value, time = facebook(50, "random", 35, 80, 0.9, 0.2)
    print(value, time)


if __name__ == "__main__":
    # random.seed(123456)
    main()
