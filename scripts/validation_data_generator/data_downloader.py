import os

import requests
from config.config import REPO_ROOT

if __name__ == "__main__":
    urls = {
        "DOID": "http://purl.obolibrary.org/obo/doid.owl",
        "SYMP": "http://purl.obolibrary.org/obo/symp.owl",
        "FMA": "http://purl.obolibrary.org/obo/fma.owl",
    }
    os.makedirs(f"{REPO_ROOT}/scripts/data/owl", exist_ok=True)

    for name, url in urls.items():
        data_path = REPO_ROOT + f"/scripts/data/owl/{name}.owl"
        if os.path.exists(data_path):
            print(f"{name}.owl already exists. Redownload? [Y/n]: ", end="")
            if input().strip().lower() != "y":
                print("Skipping.")
                continue

        print("Downloading", f"{name}.owl ... ", end="")
        req = requests.get(url)
        with open(data_path, "wb") as fw:
            fw.write(req.content)
        print("done!")
