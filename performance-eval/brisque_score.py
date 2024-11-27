import os
import json
import cv2
from brisque import BRISQUE

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'dataset')
image_folder = os.path.join(dataset_path, 'images')

image_files = [
    "166654939_80ea4ddbcc.jpg", "112178718_87270d9b4d.jpg", "103106960_e8a41d64f8.jpg",
    "1859941832_7faf6e5fa9.jpg", "1311388430_4ab0cd1a1f.jpg", "128912885_8350d277a4.jpg",
    "1271210445_7f7ecf3791.jpg", "1153704539_542f7aa3a5.jpg", "1837976956_3c45d0f9b8.jpg",
    "111537222_07e56d5a30.jpg", "1763020597_d4cc8f0f8a.jpg", "1859726819_9a793b3b44.jpg",
    "207237775_fa0a15c6fe.jpg", "1509786421_f03158adfc.jpg", "109738763_90541ef30d.jpg",
    "156967462_72db9b722c.jpg", "1424237335_b3be9920ba.jpg", "1042020065_fb3d3ba5ba.jpg",
    "1034276567_49bb87c51c.jpg", "1104133405_c04a00707f.jpg", "1262583859_653f1469a9.jpg",
    "1282392036_5a0328eb86.jpg", "1472053993_bed67a3ba7.jpg", "1579798212_d30844b4c5.jpg",
    "103205630_682ca7285b.jpg", "1525153022_06c48dbe52.jpg", "1245022983_fb329886dd.jpg",
    "1808504612_3508f3c9bb.jpg", "1683444418_815f660379.jpg", "1799271536_6e69c8f1dc.jpg",
    "1420060118_aed262d606.jpg", "1263801010_5c74bf1715.jpg", "143552829_72b6ba49d4.jpg",
    "1569562856_eedb5a0a1f.jpg", "1312954382_cf6d70d63a.jpg", "1999444757_1b92efb590.jpg",
    "2061927950_dafba5b8a3.jpg", "1009434119_febe49276a.jpg", "190965502_0b9ed331d9.jpg",
    "199809190_e3f6bbe2bc.jpg", "1184967930_9e29ce380d.jpg", "1798209205_77dbf525b0.jpg",
    "1363843090_9425d93064.jpg", "1177994172_10d143cb8d.jpg", "1260816604_570fc35836.jpg",
    "2092177624_13ab757e8b.jpg", "191003287_2915c11d8e.jpg", "1358892595_7a37c45788.jpg",
    "1100214449_d10861e633.jpg", "1397923690_d3bf1f799e.jpg"
]

results = []

brisque = BRISQUE()

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    image = cv2.imread(image_path)
    
    if image is not None:
        score = brisque.score(image)
        results.append({"image": image_file, "score": score})
    else:
        print(f"Could not read image {image_file}")

output_file = os.path.join(current_dir, 'brisque_scores.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"BRISQUE scores saved to {output_file}")