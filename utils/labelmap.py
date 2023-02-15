############
# create label map
############
import sys
import os
sys.path.append(os.path.dirname(".."))


from utils.paths import paths, files


def read_label_map(label_map_path):

    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items

def create_label_map():
    with open("labelmap.txt") as f:
        data = f.read()

    labels = []
    for line in data.split("\n"):
        if len(line) != 0:
            name, idx = line.split(":")
            labels.append({'name' : name.strip(), 'id' : int(idx.strip())})


    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write(f'\tname:\'{label["name"]}\'\n')
            f.write(f'\tid:{label["id"]}\n')
            f.write('}\n')


if __name__ == "__main__":
    create_label_map()
