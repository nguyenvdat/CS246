f = open("q2/data/browsing.txt", "r")
item2id = dict()
single_count = dict()
support_threshold = 100
# support_threshold = 1
ids = 1

# first pass
for x in f:
  line = x.split()
  for item in line:
    if not item in item2id:
      item2id[item] = ids
      single_count[ids] = 1
      ids += 1
    else:
      single_count[item2id[item]] += 1
#   break
frequent1 = {(ids, ):count for ids, count in single_count.items() if count >= support_threshold}
f.close()
id2item = {ids:item for item, ids in item2id.items()}

# second pass
# candidate_1 = [3, 4, 6, 9]
print(len(frequent1))
first2second_id = {old_id[0]: new_id + 1 for new_id, old_id in enumerate(frequent1)}
second2first_id = {new_id + 1: old_id[0] for new_id, old_id in enumerate(frequent1)}
n = len(first2second_id)
pair_count = [0 for i in range(int(n*(n-1)/2))]
frequent2 = {}
f = open("q2/data/browsing.txt", "r")
for x in f:
  line = x.split()
  for i in range(len(line) - 1):
    for j in range(i + 1, len(line)):
      old_id1 = item2id[line[i]]
      old_id2 = item2id[line[j]]
      if (old_id1, ) in frequent1 and (old_id2, ) in frequent1:
        new_id1 = first2second_id[old_id1]
        new_id2 = first2second_id[old_id2]
        first_id = new_id1 if new_id1 < new_id2 else new_id2
        second_id = new_id2 if new_id1 < new_id2 else new_id1
        pos = get_matrix_position(first_id, second_id, n)  
        pair_count[pos] += 1
        if pair_count[pos] >= support_threshold:
          frequent2[(first_id, second_id)] = pair_count[pos]
#   break
rules2 = []
for key, value in frequent2.items():
#   print((key[0], key[1]))
  item1 = id2item[second2first_id[key[0]]]
  item2 = id2item[second2first_id[key[1]]]
  rules2.append(((item1, item2), get_confidence_score((key[0],), (key[1],), frequent1, frequent2, second2first_id)))
  rules2.append(((item2, item1), get_confidence_score((key[1],), (key[0],), frequent1, frequent2, second2first_id)))
rules2.sort(key=lambda x: (-x[1], x[0]))
print(rules2[:5])
f.close()

# third pass
triple_count = {}
f = open("q2/data/browsing.txt", "r")
for x in f:
  line = x.split()
  for i in range(len(line) - 2):
    for j in range(i + 1, len(line) - 1):
      for k in range(j + 1, len(line)):
        ids1 = item2id[line[i]]
        ids2 = item2id[line[j]]
        ids3 = item2id[line[k]]
        if (ids1, ) in frequent1 and (ids2, ) in frequent1 and (ids3, ) in frequent1:
          ids1 = first2second_id[ids1]
          ids2 = first2second_id[ids2]
          ids3 = first2second_id[ids3]
        else:
          continue
        ids1, ids2, ids3 = sorted([ids1, ids2, ids3])
        if (ids1, ids2) in frequent2 and (ids2, ids3) in frequent2 and (ids1, ids3) in frequent2:
          if (ids1, ids2, ids3) in triple_count:
            triple_count[(ids1, ids2, ids3)] += 1
          else:
            triple_count[(ids1, ids2, ids3)] = 1
frequent3 = {triple_ids:count for triple_ids, count in triple_count.items() if count >= support_threshold}
rules3 = []
for triple_ids, count in frequent3.items():
  item1 = id2item[second2first_id[triple_ids[0]]]
  item2 = id2item[second2first_id[triple_ids[1]]]
  item3 = id2item[second2first_id[triple_ids[2]]]
  rules3.append(((tuple(sorted([item1, item2])), item3), 
                 get_confidence_score((triple_ids[0], triple_ids[1]), (triple_ids[2],), frequent2, frequent3)))
  rules3.append(((tuple(sorted([item2, item3])), item1), 
                 get_confidence_score((triple_ids[1], triple_ids[2]), (triple_ids[0],), frequent2, frequent3)))
  rules3.append(((tuple(sorted([item3, item1])), item2), 
                 get_confidence_score((triple_ids[2], triple_ids[0]), (triple_ids[1],), frequent2, frequent3)))
rules3.sort(key=lambda x: (-x[1], x[0]))
print(rules3[:5])

def get_matrix_position(i, j, n):
  return int((n * (n - 1) - (n - i) * (n - i + 1)) / 2 + j - i - 1)

def get_confidence_score(left_ids, right_ids, left_count_dict, joint_count_dict, id_map=None):
  if id_map:
    l_ids = tuple(sorted([id_map[i] for i in left_ids]))
  else:
    l_ids = tuple(sorted(left_ids))
  left_prob = left_count_dict[l_ids]
  joint_id = tuple(sorted(list(left_ids) + list(right_ids)))
  return joint_count_dict[joint_id]/left_prob