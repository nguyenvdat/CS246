import numpy as np
from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)


def page_rank():
    n_node = 1000
    beta = 0.8
    edges = sc.textFile('q2/data/graph-full.txt').map(lambda line: line.split()
                                                      ).map(lambda pair: (int(pair[0]), int(pair[1])))
    edges = edges.distinct()
    out_count = edges.map(lambda x: (x[0], 1))
    out_count = out_count.reduceByKey(lambda v1, v2: v1 + v2)
    edges = edges.join(out_count)
    r = [(i, 1/n_node) for i in range(1, n_node + 1)]

    for i in range(40):
        r = sc.parallelize(r)
        m = edges.join(r)  # (dest, beta * source_weight / source_out_degree)
        m = m.map(lambda x: (x[1][0][0], beta * x[1][1] / x[1][0][1]))
        m = m.reduceByKey(lambda v1, v2: v1 + v2)
        m = m.mapValues(lambda v: v + (1 - beta) / n_node)
        r = m.collect()

    r = sc.parallelize(r)
    print(r.top(5, lambda x: x[1]))
    print(r.top(5, lambda x: -x[1]))


def hits():
    n_node = 100
    lmb = 1
    mu = 1
    edges = sc.textFile('q2/data/graph-full.txt').map(lambda line: line.split()
                                                      ).map(lambda pair: (int(pair[0]), int(pair[1])))
    edges = edges.distinct()
    reversed_edges = edges.map(lambda x: (x[1], x[0]))
    h = [(i, 1) for i in range(1, n_node + 1)]
    for i in range(40):
        h = sc.parallelize(h)
        h_t = edges.join(h)  # (source, (dest, h_source))
        h_t = h_t.map(lambda x: (x[1][0], mu * x[1][1]))  # (dest, h_source)
        h_t = h_t.reduceByKey(lambda v1, v2: v1 + v2)
        max_a = h_t.max(lambda x: x[1])[1]
        a = h_t.map(lambda x: (x[0], x[1] / max_a))

        a_t = reversed_edges.join(a)  # (dest, (source, a_dest))
        a_t = a_t.map(lambda x: (x[1][0], lmb * x[1][1]))  # (source, a_dest)
        a_t = a_t.reduceByKey(lambda v1, v2: v1 + v2)
        max_h = a_t.max(lambda x: x[1])[1]
        h = a_t.map(lambda x: (x[0], x[1] / max_h))
        h = h.collect()
    h = sc.parallelize(h)
    print(h.top(5, lambda x: x[1]))
    print(a.top(5, lambda x: x[1]))
    print(h.top(5, lambda x: -x[1]))
    print(a.top(5, lambda x: -x[1]))


if __name__ == "__main__":
    page_rank()
