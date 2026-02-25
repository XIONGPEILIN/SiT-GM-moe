import os
import json
import argparse

def generate_json(features_dir):
    # Determine the subdirectories
    feat_sub = None
    label_sub = None
    L = os.listdir(features_dir)
    for name in L:
        if name.endswith('_features'):
            feat_sub = os.path.join(features_dir, name)
        elif name.endswith('_labels'):
            label_sub = os.path.join(features_dir, name)

    if not feat_sub or not label_sub:
        raise ValueError("Could not find _features and _labels subdirectories.")

    # Sort logic to match train.py exactly
    # 0_0.npy -> split('_')[0] is the batch index, x[-5] is the rank within batch?
    # Actually let's just sort them naturally or match the user's specific logic.
    # The user's earlier logic was: key=lambda x:int(x.split('_')[0])*8+int(x[-5])
    # This suggests 8 ranks per batch maybe?
    
    def sort_key(x):
        try:
            parts = x.split('_')
            # 0_0.npy -> parts[0]=0, parts[1]='0.npy'
            batch_idx = int(parts[0])
            rank_idx = int(parts[1].split('.')[0])
            return batch_idx * 1000 + rank_idx # Generic large number for rank
        except:
            return x

    features_files = sorted(os.listdir(feat_sub), key=sort_key)
    labels_files = sorted(os.listdir(label_sub), key=sort_key)
    
    # Filter out files if needed (train.py had [:-1])
    # But for a general list we might want all.
    
    data = {
        "features_dir": feat_sub,
        "labels_dir": label_sub,
        "features_files": features_files,
        "labels_files": labels_files
    }
    
    output_path = os.path.join(features_dir, "file_list.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Generated {output_path} with {len(features_files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()
    generate_json(args.dir)
