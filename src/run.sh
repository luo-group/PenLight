python train.py --experiment cath-demo --model gatv2 --aa_embed protT5 --batch_hard --task cath --parallel
python predict.py --model cath-GATv2 --task cath

python train.py --experiment ec-demo --model gatv2.01 --aa_embed esm --batch_hard --task ec --parallel
python predict.py --model ec-GATv2.01 --task ec