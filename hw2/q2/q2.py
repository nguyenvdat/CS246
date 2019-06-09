from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf()
sc = SparkContext(conf=conf)

# get points rdd from file
points_file = open('data.txt', 'r')
points_lines = points_file.readlines()
points = [(i, np.array([float(v) for v in l.split()]))
          for i, l in enumerate(points_lines)]
points = sc.parallelize(points)

# get centroids rdd from file
centroids_file = open('c2.txt', 'r')
centroids_lines = centroids_file.readlines()
centroids = [(i, np.array([float(v) for v in c.split()]))
             for i, c in enumerate(centroids_lines)]
centroids = sc.parallelize(centroids)

# k-means with Euclidean distance
costs = []
for i in range(20):
    points_centroids = points.cartesian(centroids)
    points_centroids_distance = points_centroids.map(lambda x: (
        x[0][0], (x[0][1], x[1][0], np.sum((x[0][1] - x[1][1]) ** 2))))
    # (point, (coordinate, centroid, d))
    points_centroids_distance.take(3)

    assigned_centroids = points_centroids_distance.reduceByKey(
        lambda v1, v2: v1 if v1[-1] < v2[-1] else v2)
    # (point, (coordinate, centroid, d))
    d = assigned_centroids.map(lambda x: x[1][-1])
    cost = d.reduce(lambda x, y: x + y)
    print(cost)
    costs.append(cost)
    assignment = assigned_centroids.map(lambda x: (x[0], x[1][1]))

    assigned_centroids = assigned_centroids.map(
        lambda x: (x[1][1], (x[1][0], 1)))
    assigned_centroids = assigned_centroids.reduceByKey(
        lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1]))
    assigned_centroids = assigned_centroids.map(
        lambda x: (x[0], x[1][0]/x[1][1])).collect()
    centroids = sc.parallelize(assigned_centroids)

# k-means with Manhattan distance
costs = []
for i in range(20):
    points_centroids = points.cartesian(centroids)
    points_centroids_distance = points_centroids.map(lambda x: (
        x[0][0], (x[0][1], x[1][0], np.sum(np.abs(x[0][1] - x[1][1])))))
    # (point, (coordinate, centroid, d))
    points_centroids_distance.take(3)

    assigned_centroids = points_centroids_distance.reduceByKey(
        lambda v1, v2: v1 if v1[-1] < v2[-1] else v2)
    # (point, (coordinate, centroid, d))
    d = assigned_centroids.map(lambda x: x[1][-1])
    cost = d.reduce(lambda x, y: x + y)
    print(cost)
    costs.append(cost)
    assignment = assigned_centroids.map(lambda x: (x[0], x[1][1]))

    assigned_centroids = assigned_centroids.map(
        lambda x: (x[1][1], (x[1][0], 1)))
    assigned_centroids = assigned_centroids.reduceByKey(
        lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1]))
    assigned_centroids = assigned_centroids.map(
        lambda x: (x[0], x[1][0]/x[1][1])).collect()
    centroids = sc.parallelize(assigned_centroids)

# plot cost vs iteration
plt.plot(range(1, 21), costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.xticks(np.arange(1, 21, 1.0))
plt.title('Cost over iteration of c2.txt centroids with Manhattan distance', y=1.1)
plt.show()

# get percentage change in cost
(costs[0] - costs[9])/costs[0]
