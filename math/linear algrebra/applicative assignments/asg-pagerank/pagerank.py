import numpy as np
from numpy import linalg as la

np.set_printoptions(precision=3)



class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].                

        Examples
        ========
        >>> A = np.array([[0, 0, 0, 0],[1, 0, 1, 0],[1, 0, 0, 1],[1, 0, 1, 0]])
        >>> G = DiGraph(A, labels=['a','b','c','d'])
        >>> G.A_hat
        array([[0.   , 0.25 , 0.   , 0.   ],
               [0.333, 0.25 , 0.5  , 0.   ],
               [0.333, 0.25 , 0.   , 1.   ],
               [0.333, 0.25 , 0.5  , 0.   ]])
        >>> steady_state_1 = G.linsolve()
        >>> { k: round(steady_state_1[k],3) for k in steady_state_1}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_2 = G.eigensolve()
        >>> { k: round(steady_state_2[k],3) for k in steady_state_2}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_3 = G.itersolve()
        >>> { k: round(steady_state_3[k],3) for k in steady_state_3}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> get_ranks(steady_state_3)
        ['c', 'b', 'd', 'a']
        """
        # Store number of nodes (size of graph)
        num_of_nodes = A.shape[0]
        
        # Set default labels if none provided
        if labels is None:
            self.labels = list(range(num_of_nodes))
        else:
            self.labels = labels
        
        # Copy to modify
        A_modified = A.astype(float).copy()
        
        # Find sinks (columns with sum 0) and set them to 1
        col_sums = np.sum(A_modified, axis=0)
        for j in range(num_of_nodes):
            if col_sums[j] == 0:  # absorbing state
                A_modified[:, j] = 1.0  # Set column to 1
        
        # Normalize columns to create A_hat
        # Each column should sum to 1
        col_sums = np.sum(A_modified, axis=0)
        self.A_hat = A_modified / col_sums


    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # The PageRank equation: (I - ε*A_hat) * x = ((1-ε)/n) * 1_n
        num_of_nodes = self.A_hat.shape[0]
        
        # Identity matrix
        I = np.eye(num_of_nodes)
        
        # Create coefficient matrix: I - ε*A_hat
        A_coeff = I - epsilon * self.A_hat
        
        # Create rds: ((1-ε)/n) * 1_n
        b = ((1 - epsilon) / num_of_nodes) * np.ones(num_of_nodes)
        
        # Solve the linear system
        x = la.solve(A_coeff, b)
        
        # Dictionary mapping labels to PageRank values
        return {self.labels[i]: float(x[i]) for i in range(num_of_nodes)}


    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        num_of_nodes = self.A_hat.shape[0]
        
        # Build the Google matrix P = ε*A_hat + (1-ε)/n * ones_matrix
        google_matrix = epsilon * self.A_hat + ((1 - epsilon) / num_of_nodes) * np.ones((num_of_nodes, num_of_nodes))
        
        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = la.eig(google_matrix)
        
        # Find the eigenvalue closest to 1 (the dominant eigenvalue)
        idx = np.argmax(np.real(eigenvalues))
        
        # Extract the corresponding eigenvector
        pagerank_vector = eigenvectors[:, idx]
        
        # Normalize so entries sum to 1 and make positive
        pagerank_vector = np.abs(pagerank_vector)
        pagerank_vector = pagerank_vector / np.sum(pagerank_vector)
        
        # Return as dictionary mapping labels to PageRank values
        return {self.labels[i]: float(pagerank_vector[i]) for i in range(num_of_nodes)}
        


    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        num_of_nodes = self.A_hat.shape[0]
        x = np.ones(num_of_nodes) / num_of_nodes  # Start with uniform distribution

        for _ in range(maxiter):
            # using 
            x_new = epsilon * (self.A_hat @ x) + ((1 - epsilon) / num_of_nodes) * np.ones(num_of_nodes)
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
          

        return {self.labels[i]: float(x[i]) for i in range(num_of_nodes)}



def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """    
        # Sort by PageRank descending, then lexicographically ascending for ties
    return sorted(d, key=lambda k: (-d[k], k))
    


# Task 2
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks().

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.

    Examples
    ========
    >>> print(rank_websites()[0:5])
    ['98595', '32791', '28392', '77323', '92715']
    """
    # Read the file and build the set of all unique page IDs
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    edges = []
    nodes = set()
    for line in lines:
        ids = line.split("/")
        src = ids[0]
        nodes.add(src)
        for dst in ids[1:]:
            nodes.add(dst)
            edges.append((src, dst))
    labels = sorted(nodes)
    label_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)

    # Build adjacency matrix: A[i, j] = 1 if j links to i
    A = np.zeros((n, n))
    for src, dst in edges:
        i = label_idx[dst]
        j = label_idx[src]
        A[i, j] = 1
        
    # Make DiGraph and compute PageRank using already impemented itersolve (from task 1)
    G = DiGraph(A, labels)
    pagerank = G.itersolve(epsilon)
    return get_ranks(pagerank)


# Task 3
def rank_uefa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.

    Examples
    ========
    >>> rank_uefa_teams("psh-uefa-2018-2019.csv",0.85)[0:5]
    ['Liverpool', 'Ath Madrid', 'Paris SG', 'Genk', 'Barcelona']
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    edges = []
    nodes = set()
    for line in lines:
        parts = line.split(",")
        home, away, home_goals, away_goals = parts[0], parts[1], parts[2], parts[3]
        if home_goals == away_goals:
            continue  # ignore draws
        nodes.add(home)
        nodes.add(away)
        if home_goals > away_goals:
            edges.append((home, away))  # away lost to home
        else:
            edges.append((away, home))  # home lost to away
    labels = sorted(nodes)
    label_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)

    # Build adjacency matrix: A[i, j] = number of times j lost to i
    A = np.zeros((n, n))
    for winner, loser in edges:
        i = label_idx[winner]
        j = label_idx[loser]
        A[i, j] += 1
    # Make DiGraph and compute PageRank using already implemented itersolve
    G = DiGraph(A, labels)
    pagerank = G.itersolve(epsilon)
    sorted_teams = get_ranks(pagerank)
    return sorted_teams

if __name__ == "__main__":
    import doctest
    doctest.testmod()
