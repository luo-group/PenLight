import json, os, argparse
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', '-d', type=str)
    parser.add_argument('--task', '-t', type=str, choices=['cath', 'ec'])
    
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.dir, 'splits_json'), exist_ok=True)
    all_data_path = os.path.join(args.dir, 'cath_data_S100.json' if args.task=='cath' else 'all_data.json')
    with open(all_data_path) as f:
        all_data = json.load(f)
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith('_ids.txt'):
                fid = file.split('_')[0]
                with open(os.path.join(root, file)) as f:
                    indices = f.read().strip().split()
                split_data = {}
                for idx in indices:
                    split_data[idx] = all_data[idx]
                with open(os.path.join(args.dir, 'splits_json', f'{fid}.json'), 'w') as f:
                    json.dump(split_data, f)
                print(f"Generated {os.path.join(args.dir, 'splits_json', f'{fid}.json')} with {len(split_data)} entries")
    with open(os.path.join(args.dir, 'all_data.fasta'), 'w') as f:
        for k, v in all_data.items():
            f.write(f'>{k}\n{v}\n') 
    print(f"Generated fasta file {os.path.join(args.dir, 'all_data.fasta')}")
    
    