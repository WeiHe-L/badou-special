import os

photos = os.listdir("./test_image/")
#print(photos[1])

with open("./image_dataset/test_dataset.txt", "w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name == "cat":
            f.write(photo + ";0\n")
        elif name == "dog":
            f.write(photo + ";1\n")
f.close()