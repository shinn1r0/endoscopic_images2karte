SCRIPTS_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=$(dirname $SCRIPTS_DIR)
PAPER_DIR=$PROJECT_DIR/paper
cd $PAPER_DIR
latexmk -f main.tex
latexmk -c
cd $SCRIPT_DIR
