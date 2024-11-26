# 環境搭建
conda deactivate
conda env remove -n trade
conda create -n  trade
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install pandas
conda install matplotlib
conda install anaconda::scikit-learn