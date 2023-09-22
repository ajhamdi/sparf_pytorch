def read_list(path):
    with open(path, 'r') as f:
        ids = [int(x[4:]) for x in f.readlines()]
    return ids

mvsnet_ids = []
mvsnet_ids.extend(read_list('DTU/mvsnet_train.lst'))
mvsnet_ids.extend(read_list('DTU/mvsnet_val.lst'))
mvsnet_ids.extend(read_list('DTU/mvsnet_test.lst'))
mvsnet_ids = sorted(mvsnet_ids)

new_val_ids = read_list('DTU/new_val.lst')
print(new_val_ids)

manual_exclude = [1, 2, 7, 29, 39, 51, 56, 57, 58, 83, 111, 112, 113, 115, 116, 117]


remaining = [i for i in range(1, 129) if i in mvsnet_ids and i not in new_val_ids and i not in manual_exclude]
print('Extracted', len(remaining))
print(remaining)

txt = '\n'.join(['scan' + str(i) for i in remaining])
with open('DTU/new_train.lst', 'w') as f:
    f.write(txt)

