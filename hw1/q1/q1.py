from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile('data/soc-LiveJournal1Adj.txt')
friend_list = lines.map(lambda l: (l.split('\t')[0], l.split('\t')[1].split(','))) # 1 -> [2, 3, 4]
friend_pairs = friend_list.flatMap(lambda pair: [(f, pair[0]) for f in pair[1]]) # [2, 1] [3, 1] [4, 1]
friend_pair_list = friend_pairs.join(friend_list)
mutual_pairs = friend_pair_list.flatMap(lambda pair: [((pair[1][0], f), 1) for f in pair[1][1] if pair[1][0] != f])
mutual_pairs_count = mutual_pairs.reduceByKey(lambda v1, v2: v1 + v2) # [((this_person, another_person), number_mutual_friends)]
fr_pairs = friend_list.flatMap(lambda pair: [((pair[0], f), 1) for f in pair[1]])
mutual_pairs_count = mutual_pairs_count.subtractByKey(fr_pairs)

recom_list = mutual_pairs_count.map(lambda pair: (pair[0][0], (pair[1], pair[0][1]))) #[(this_person, (another_person, number_mutual_friends))]
all_people = friend_list.map(lambda pair: (pair[0], 1)) 
recom_list = recom_list.groupByKey().fullOuterJoin(all_people) # [(this_person, (iterable, 1))]

f = open('recom.txt', "w+")
for pair in recom_list.sortBy(lambda pair: int(pair[0])).collect():
  if pair[1][0] is not None: # have mutual friends with someone
    f.write("%s\t%s\r\n" % (pair[0], ",".join([p[1] for p in sorted(pair[1][0], key=lambda x: (-x[0], int(x[1])))][:10])))
  else:
    f.write("%s\r\n" % (pair[0]))
f.close()