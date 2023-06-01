import json
import os
# from collections import defaultdict


def filenameParser(filename: str) -> tuple[str, str, str]:
    filename = filename[:-4]
    date, time, post_id = filename.split('_')
    return date, time, post_id


# post_index = defaultdict(list)
post_index = {}

for year_dir in os.listdir("./HatePolitics/"):
    for filename in os.listdir(os.path.join("./HatePolitics/", year_dir)):
        date, time, post_id = filenameParser(filename)
        post_index[post_id] = f"./HatePolitics/{year_dir}/{filename}"
    
json.dump(post_index, open("post_index.json", "w"))
