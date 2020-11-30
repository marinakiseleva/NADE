export DATASETSPATH=/Users/marina/Documents/PhD/research/astro_research/data/testing/
export RESULTSPATH=./output
export PYTHONPATH=./buml:$PYTHONPATH
mkdir -p output
echo $DATASETSPATH
echo $RESULTSPATH
python2 originalNADE.py --theano --form MoG --dataset Unspecified_IaX.hdf5 --training_route /folds/1/training/\(1\|2\|3\|4\|5\|6\|7\|8\) --validation_route /folds/1/training/9 --test_route /folds/1/tests/.* --samples_name data --hlayers 2  --layerwise --lr 0.02 --wd 0.02 --n_components 10 --epoch_size 100 --momentum 0.9 --units 100  --pretraining_epochs 5 --validation_loops 20 --epochs 10 --normalize --batch_size 100 --show_training_stop True