# 6.s058-Project
6.s058 Final Project: "Manga Characters Identification with Contrastive Learning"

We use generative AI Claude Haiku 4.5 and Gemini Pro 3.1 to assist with partly of implementation.

Due to restricted access to the dataset, please see information about access to the dataset on
https://manga109.github.io/manga109-project-website/en/index.html

## Set up

Install all packages as defined in `requirements.txt`.

## Training
- set parameters in `params.py`
- execute 

```
python contrastive.py
```

## Evaluation
For the our trained model please download from [Our dropbox folder](https://www.dropbox.com/scl/fo/4cxr6musyidm9zm6sks6q/APdjBkLvuXtqS3Z1W5WBFPs?rlkey=fn8upm79gk766u4qblf8pr6x7&st=rlb2p3xu&dl=0).

The model name must in form `simclr-x.pt` where `x` is a number.

- Evaluate the encoder only
```
python predict_compare.py
```
Edit the numbers in line `for trial_no in [...]` to choose the model to evaluate
- Evaluate the entire pipeline
```
python predict_dataset.py
```
Edit numbers in line `trial_no = ...`
The output will be recorded in
`acc-{trial_no}-yolo-charbase.csv`
