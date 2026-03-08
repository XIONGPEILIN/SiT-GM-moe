import json
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_shape(path):
    return np.load(path, mmap_mode='r').shape[0]


def convert(features_dir="/home/yanai-lab/xiong-p/SiT-GM-moe/imagenet_feature"):
    json_path = os.path.join(features_dir, "file_list.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    features_files = data['features_files']
    labels_files = data['labels_files']
    base_features_dir = data['features_dir']
    base_labels_dir = data['labels_dir']

    out_features = os.path.join(features_dir, "merged_features.npy")
    out_labels = os.path.join(features_dir, "merged_labels.npy")

    if os.path.exists(out_features) and os.path.exists(out_labels):
        print("Merged files already exist.")
        return

    first_f = np.load(os.path.join(base_features_dir, features_files[0]))
    first_l = np.load(os.path.join(base_labels_dir, labels_files[0]))

    print("Pre-scanning to get exact offsets using multiple threads...")
    shapes = [0] * len(features_files)

    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_idx = {
            executor.submit(get_shape, os.path.join(base_features_dir, f)): i
            for i, f in enumerate(features_files)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(features_files), desc="Scanning"):
            idx = future_to_idx[future]
            shapes[idx] = future.result()

    total = sum(shapes)
    print(f"Total precise samples: {total}")
    shape_f = (total,) + first_f.shape[1:]
    shape_l = (total,) + first_l.shape[1:]

    print(
        f"Allocating uninitialized memmap for Features: {shape_f} and Labels: {shape_l}")
    fm = np.lib.format.open_memmap(
        out_features, mode='w+', dtype=first_f.dtype, shape=shape_f)
    lm = np.lib.format.open_memmap(
        out_labels, mode='w+', dtype=first_l.dtype, shape=shape_l)

    # Compute starting offsets
    offsets = [0] * len(features_files)
    curr = 0
    for i, s in enumerate(shapes):
        offsets[i] = curr
        curr += s

    print("Writing merged files using multiple threads...")

    def process_file(i):
        fn = features_files[i]
        ln = labels_files[i]
        f_path = os.path.join(base_features_dir, fn)
        l_path = os.path.join(base_labels_dir, ln)

        f_arr = np.load(f_path)
        l_arr = np.load(l_path)

        offset = offsets[i]
        n = f_arr.shape[0]
        # Memory map writes perfectly support concurrent non-overlapping updates
        fm[offset:offset+n] = f_arr
        lm[offset:offset+n] = l_arr

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_file, i)
                   for i in range(len(features_files))]
        for future in tqdm(as_completed(futures), total=len(features_files), desc="Writing"):
            future.result()

    fm.flush()
    lm.flush()
    print("Done! the new loader will be lightning fast.")


if __name__ == "__main__":
    convert()
