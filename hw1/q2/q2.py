f = open("q2/data/browsing.txt", "r")
item2id_count = dict()
support_threshold = 100
# support_threshold = 1
id_count = 1

# first pass
for x in f:
    line = x.split()
    for item in line:
        if not item in item2id_count:
            item2id_count[item] = [id_count, 1]
            id_count += 1
        else:
            item2id_count[item][1] += 1
#   break
frequent1 = {id_count[0]: id_count[1] for item, id_count in item2id_count.items(
) if id_count[1] >= support_threshold}
f.close()
# # second pass
# candidate_1 = [3, 4, 6, 9]
print(len(frequent1))
first2second_id = {old_id: new_id +
                   1 for new_id, old_id in enumerate(frequent1)}
second2first_id = {new_id + 1: old_id for new_id,
                   old_id in enumerate(frequent1)}
n = len(first2second_id)
pair_count = [0 for i in range(int(n*(n-1)/2))]
frequent2 = {}
f = open("q2/data/browsing.txt", "r")
for x in f:
    line = x.split()
    for i in range(len(line) - 1):
        for j in range(i + 1, len(line)):
            old_id1 = item2id_count[line[i]][0]
            old_id2 = item2id_count[line[j]][0]
            if old_id1 in frequent1 and old_id2 in frequent1:
                new_id1 = first2second_id[old_id1]
                new_id2 = first2second_id[old_id2]
                first_id = new_id1 if new_id1 < new_id2 else new_id2
                second_id = new_id2 if new_id1 < new_id2 else new_id1
                pos = get_matrix_position(first_id, second_id, n)
#         print(first_id)
#         print(second_id)
#         print(pos)
#         print()
                pair_count[pos] += 1
                if pair_count[pos] >= support_threshold:
                    frequent2[(first_id, second_id)] = pair_count[pos]
#   break
rules2 = []
for key, value in frequent2.items():
    #   print((key[0], key[1]))
    rules2.append(((key[0], key[1]), get_confidence_score(
        key[0], key[1], frequent1, frequent2, second2first_id)))
    rules2.append(((key[1], key[0]), get_confidence_score(
        key[1], key[0], frequent1, frequent2, second2first_id)))
rules2.sort(key=lambda x: x[1], reverse=True)
print(rules2[:5])


def get_matrix_position(i, j, n):
    return int((n * (n - 1) - (n - i) * (n - i + 1)) / 2 + j - i - 1)


def get_confidence_score(left_id, right_id, left_count_dict, joint_count_dict, id_map):
    l_id = (id_map[i] for i in left_id) if type(
        left_id) == tuple else id_map[left_id]
    l_id = tuple(sorted(l_id)) if type(l_id) == tuple else l_id
    left_prob = left_count_dict[l_id]
    l_id = list(left_id) if type(left_id) == tuple else [left_id, ]
    r_id = list(right_id) if type(right_id) == tuple else [right_id, ]
    joint_id = tuple(sorted(l_id + r_id))
#   print(joint_id)
    return joint_count_dict[joint_id]/left_prob
