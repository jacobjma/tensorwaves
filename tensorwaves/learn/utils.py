import numpy as np


def nms(positions, probabilities, radius=2):
    from sklearn.neighbors import NearestNeighbors
    order = np.argsort(-probabilities)

    positions = positions[order]
    probabilities = probabilities[order]

    nn = NearestNeighbors().fit(positions)
    _, neighbors = nn.radius_neighbors(positions, radius=radius)

    accepted = np.zeros_like(order, dtype=bool)
    suppressed = np.zeros_like(order, dtype=bool)

    for i in range(len(positions)):
        if (not suppressed[i]):
            order = np.argsort(probabilities[neighbors[i]])
            n = neighbors[i][order]
            suppressed[n] = True
            if n[-1] == i:
                accepted[i] = True

    return positions[accepted], probabilities[accepted]
