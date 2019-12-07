SCRIPTS_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=$(dirname $SCRIPTS_DIR)
SRC_DIR=$PROJECT_DIR/images2karte
python $SRC_DIR/train.py -b 50 -e 30 -d dataset2 -m densenet_121
python $SRC_DIR/train.py -b 50 -e 30 -d dataset2 -m densenet_121_expansion
python $SRC_DIR/train.py -b 25 -e 30 -d dataset2 -m densenet_161
python $SRC_DIR/train.py -b 25 -e 30 -d dataset2 -m densenet_161_expansion
cp -r $OUTPUTS_DIR/densenet_161_expansion* $GOOGLEDRIVE/outputs/
python $SRC_DIR/train.py -b 5 -e 30 -d dataset2 -m lrcn -s -i 60
python $SRC_DIR/train.py -b 3 -e 30 -d dataset2 -m resnet3d -s -i 60

python $SRC_DIR/test.py -r densenet121_e_p02 --image_threshold 0.5 --label_threshold 0.3 -v
python $SRC_DIR/test.py -r densenet121_e_p02 --image_threshold 0.1 --label_threshold 0.0 -v
python $SRC_DIR/test.py -r densenet121_e_p02 -i --image_threshold 0.5 --label_threshold 0.3 -v
