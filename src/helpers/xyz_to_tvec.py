import yaml

height = float(6)
width = float(6)

translation_vectors = {}

ids = int(input("id len: "))
for i in range(ids):
    print("id " + str(i + 1))
    x = input("x: ").replace(" in.", "")
    y = input("y: ").replace(" in.", "")
    z = input("z: ").replace(" in.", "")

    x = float(x)
    y = float(y)
    z = float(z)

    # Top left, top right, bottom left, bottom right
    translation_vectors[i + 1] = {
        "pos": {
            "topLeft": [x + height, y, z],
            "topRight": [x + height, y + width, z],
            "bottomLeft": [x, y, z],
            "bottomRight": [x, y + width, z]
        },
        "size": {
            "height": height,
            "width": width
        }
    }

with open("tvecs.yml", "w") as f:
    yaml.dump(translation_vectors, f)
