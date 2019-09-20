# ROC Curve Demo
Matthew Epland, PhD  
[Komodo Health](https://www.komodohealth.com/)  
[phy.duke.edu/~mbe9](http://www.phy.duke.edu/~mbe9)  

A demonstration of TPR vs FPR and Precision vs Recall ROC curves on a synthetic dataset with XGBoost  

## Cloning the Repository
ssh  
```bash
git clone --recurse-submodules git@github.com:mepland/roc_curve_demo.git
```

https  
```bash
git clone --recurse-submodules https://github.com/mepland/roc_curve_demo.git
```

## Installing Dependencies
It is recommended to work in a [python virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to avoid clashes with other installed software.  
```bash
python -m venv ~/.venvs/newenv
source ~/.venvs/newenv/bin/activate
pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter lab roc_curve_demo.ipynb
```
