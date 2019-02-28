import numpy as np
import itertools
from scipy.spatial import Voronoi


def voronoi_centroids(points):
    def area_centroid(points):
        points = np.vstack((points, points[0]))

        A = 0
        C = np.zeros(2)
        for i in range(0, len(points) - 1):
            s = points[i, 0] * points[i + 1, 1] - points[i + 1, 0] * points[i, 1]
            A = A + s
            C = C + (points[i, :] + points[i + 1, :]) * s

        return (1 / (3. * A)) * C

    vor = Voronoi(points)

    centroids = []

    for i, point in enumerate(points):
        if all(np.array(vor.regions[vor.point_region[i]]) > -1):
            vertices = vor.vertices[vor.regions[vor.point_region[i]]]
            centroids.append(area_centroid(vertices))

    return np.array(centroids)


def relax(points, box, num_iter=1, bc='periodic', wrap=True):
    def mirror(points, box):

        new_points = points.copy()

        for i, j in zip([0, 0, 1, 1], [2, 0, 2, 0]):
            tmp_points = points.copy()
            tmp_points[:, i] = j * np.array(box)[i] - tmp_points[:, i]
            new_points = np.concatenate((new_points, tmp_points))

        return new_points

    def repeat(points, box):

        new_points = points.copy()

        for direction in set(itertools.combinations([-1, 0, 1] * 2, 2)):
            if direction != (0, 0):
                new_points = np.concatenate((new_points, points + direction * np.array(box)))

        return new_points

    N = len(points)

    for i in range(num_iter):
        if box is not None:
            if bc == 'periodic':
                points = repeat(points, box)
            elif bc == 'mirror':
                points = mirror(points, box)
            else:
                raise NotImplementedError('Boundary condition {0} not recognized'.format(bc))

        centroids = voronoi_centroids(points)
        points = centroids[:N]

        if wrap:
            points[:, 0] = points[:, 0] % box[0]
            points[:, 1] = points[:, 1] % box[1]

    return points
