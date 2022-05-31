## Server Run Command

```
#Run Code
cd /home/ajahin/Keyword-Extractor/TermExtractor
nohup python3 TermExtractor-main.py>output.txt &

# See Output
tail -f ~/output.txt

#See List
ps xw
ps -aux | grep ajahin

#Kill a process

kill -9 <PId>
```

## Install conda package command-
```console
conda env remove -n BiLstm -y
#conda env create -f environment.yml
conda env update --file environment.yml
activate BiLstm

#pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en_core_web_sm
#deactivate
```
And then activate the `BiLstm` environment for the code.

Install `spacy` packagge with this command-

    python -m spacy download en_core_web_sm

## Required pip packages for `python`=`3.6.5`:

Can be found in [this](./environment.yml) file.