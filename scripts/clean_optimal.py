folder_name = input("Dataset name")
paths = list()
with open("outputs/" + folder_name + "/" + "path_optimal.txt", "r") as path:
    for line in path.readlines():
        temp = [int(x) for x in line.rstrip().split(",")]
        paths.extend(temp)

paths = [str(x - 1) for x in paths]
paths.append(paths[0])
# paths.reverse()

with open("outputs/" + folder_name + "/" + "path_optimal.txt", "w+") as path:
    path.write(",".join(paths))
