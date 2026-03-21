import matplotlib.pyplot as plt

def farey(n):
    """
    Returns the Farey sequence of order n as a list of (numerator, denominator) tuples.
    The list represents reduced fractions a/b such that 0 <= a <= b <= n,
    ordered by increasing magnitude.
    """
    if n <= 0:
        return []
        
    farey_seq = [(0, 1)]
    
    a, b = 0, 1
    c, d = 1, n
    
    farey_seq.append((c, d))

    while not (c == 1 and d == 1):
        k = (n + b) // d
        p = k * c - a
        q = k * d - b
        a, b = c, d
        c, d = p, q
        farey_seq.append((c, d))
        
    return farey_seq


def thomae(n):
    """
    Plots the Farey approximant of order n.
    """
    plt.rcParams.update({'font.size': 20})
    seq = farey(n)
    xs = [a / b for a, b in seq]
    ys = [1 / b for a, b in seq]
    
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o', linestyle='-', markersize=4)
    # plt.title(f'Farey Approximant of Order {n}')
    plt.xlabel(
        rf"$\mathbf{{{n}}}$-Farey Approximant to Thomae’s function")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.show()


def thomae_matrix(matrix):
    """
    Plots a matrix of Farey approximants.
    matrix: list of lists of integers.
    Saves the result to thomae_array.png.
    """
    rows = len(matrix)
    if rows == 0:
        return
    cols = len(matrix[0])
    
    # squeeze=False ensures axes is always a 2D array
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    
    # Ensure font size matches thomae(n)
    plt.rcParams.update({'font.size': 20})
        
    for i in range(rows):
        for j in range(cols):
            n = matrix[i][j]
            ax = axes[i][j]
            
            seq = farey(n)
            xs = [a / b for a, b in seq]
            ys = [1 / b for a, b in seq]
            
            ax.plot(xs, ys, marker='o', linestyle='-', markersize=4)
            
            # Using the same label style as thomae(n)
            ax.set_xlabel(rf"$\mathbf{{{n}}}$-Farey Approximant")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.grid(True)

    plt.tight_layout()
    plt.savefig('thomae_array.png')
    plt.close()
