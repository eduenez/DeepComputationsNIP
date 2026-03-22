import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def farey_scatter(n):
    """
    Returns the x and y coordinates for a scatter plot of the Farey sequence of order n.
    x = a/b and y = 1/b for each fraction a/b in the Farey sequence
    """
    seq = farey(n)
    a, b = seq[0]  # Start with the first fraction (0/1)
    xs, ys = [0], [1]  # Start with the point (0, 1) for 0/1

    for c, d in seq[1:]:  # Skip the first fraction since it's already added
        xs += [(a + c) / (b + d), c / d]
        ys += [0, 1 / d]
        a, b = c, d  # Move to the next fraction
    return xs, ys

def thomae(n):
    """
    Plots the Farey approximant of order n.
    """
    plt.rcParams.update({'font.size': 20})
    xs, ys = farey_scatter(n)

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, linestyle='-', markersize=4)  # marker='o',
    # plt.title(f'Farey Approximant of Order {n}')
    plt.xlabel(
        rf"$\mathbf{{{n}}}$-Farey Approximant to Thomae’s function")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.savefig('thomae.png')
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
    _, axes = plt.subplots(
        rows, cols, 
        figsize=(5 * cols, 5 * rows), 
        squeeze=False)

    # Ensure font size matches thomae(n)
    plt.rcParams.update({'font.size': 20})

    for i in range(rows):
        for j in range(cols):
            n = matrix[i][j]
            ax = axes[i][j]
            xs, ys = farey_scatter(n)
            ax.plot(xs, ys, marker='o', linestyle='-', markersize=4)
            # Using the same label style as thomae(n)
            ax.set_xlabel(rf"$\mathbf{{{n}}}$-Farey Approximant")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            # ax.grid(True)

    plt.tight_layout()
    plt.savefig('thomae_array.png')
    plt.close()


def thomae_animation(nlist):
    """
    Creates an animated GIF of Farey approximants for n in nlist.
    Removes x-labels from the plots.
    Saves the result to thomae_animation.gif.
    """
    if not nlist:
        return

    # Use the same font size and figure size as thomae(n)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(n):
        ax.clear()
        xs, ys = farey_scatter(n)
        ax.plot(xs, ys, linestyle='-', markersize=4)  # marker='o', 
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        # ax.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=nlist, repeat=True)
    # Using pillow writer is explicitly supported in matplotlib > 3.3
    # If not available, it might fallback or fail, but pillow is widespread.
    ani.save('thomae_animation.gif', writer='pillow', fps=2)

    plt.close()


def dirichlet_animation(temperatures, large_n=32):
    """
    Creates an animated GIF of approximants to Dirichlet's function for s in the list `temperatures`.
    Saves the result to `dirichlet_animation.gif`.
    """
    if not temperatures:
        return

    # Use the same font size and figure size as thomae(n)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(10, 6))
    xs, ys = farey_scatter(large_n)

    def update(s):
        ax.clear()
        yscaled = [y ** s for y in ys]
        ax.plot(xs, yscaled, linestyle='-', markersize=4)  # marker='o',
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        # ax.grid(True)

    ani = animation.FuncAnimation(
        fig, update, frames=temperatures, repeat=True)
    # Using pillow writer is explicitly supported in matplotlib > 3.3
    # If not available, it might fallback or fail, but pillow is widespread.
    ani.save('dirichlet_animation.gif', writer='pillow', fps=2)

    plt.close()
