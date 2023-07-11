import numpy as np
import matplotlib.pyplot as plt

sigma = 0.05
n = 10
lam = 2 * 10 ** (-1)
iter_limit = 10**3
q_tab = n * [1]


# calculates the 2-coordinates vector of derivative with respect of [x_i,y_i]
def dP_i(pos, i):
    res = np.array([0.0, 0.0])
    n = len(pos)
    for j in range(n):
        if j != i:
            res_x = (
                q_tab[j]
                * (pos[i][0] - pos[j][0])
                * (np.linalg.norm(pos[i] - pos[j])) ** (-3)
            )
            res_y = (
                q_tab[j]
                * (pos[i][1] - pos[j][1])
                * (np.linalg.norm(pos[i] - pos[j])) ** (-3)
            )

            res += np.array([float(res_x), float(res_y)])
    return np.array(q_tab[i] * res)


def grad_P(pos):  # this is actually a n x 2 matrix
    n = len(pos)
    return np.array([dP_i(pos, i) for i in range(n)])


def dP_i(pos, i):  # gradient with a gaussian potential
    res = np.array([0.0, 0.0])
    n = len(pos)
    for j in range(n):
        if j != i:
            res_x = (
                -1
                / sigma
                * q_tab[j]
                * (pos[i][0] - pos[j][0])
                * (
                    np.exp(
                        -(((pos[i][0] - pos[j][0])) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                        / sigma
                    )
                )
            )
            res_y = (
                -1
                / sigma
                * q_tab[j]
                * (pos[i][1] - pos[j][1])
                * (
                    np.exp(
                        -(((pos[i][0] - pos[j][0])) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                        / sigma
                    )
                )
            )
            res += np.array([float(res_x), float(res_y)])
    return np.array(q_tab[i] * res)


def simple_proj(v):  # projection of a 2D vector on the unit disk
    if np.linalg.norm(v) == 0:
        return v
    else:
        return min(1, 1 / np.linalg.norm(v)) * np.array(v)


def proj_pos(pos):
    return np.array([simple_proj(v) for v in pos])


def gradient_descent(initial_positions):
    pos_current = initial_positions
    iter = 0
    while iter < iter_limit:
        pos_current = proj_pos(pos_current - lam * grad_P(pos_current))
        iter += 1
    return pos_current


def plot_circle_and_points(coordinates):
    fig, ax = plt.subplots()

    # Plotting the circle
    circle = plt.Circle((0, 0), radius=1, edgecolor="black", facecolor="none")
    ax.add_artist(circle)

    # Plotting the points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], color="red", label="Candles")

    # Set plot limits and labels
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")

    plt.legend()
    plt.show()


def generate_points_on_disk(n):
    # Generate uniformly distributed angles
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Calculate radial distance from the center of the disk
    radius = np.sqrt(np.random.uniform(size=n))

    # Convert polar coordinates to Cartesian coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Combine x and y coordinates to obtain the points
    points = np.column_stack((x, y))

    return points


points = generate_points_on_disk(n)
final_points = gradient_descent(points)
plot_circle_and_points(points)
plot_circle_and_points(final_points)
