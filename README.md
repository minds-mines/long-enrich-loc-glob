# Code for "Longitudinal Enrichment of Imaging Biomarker Representations for Improved Alzheimer’s Disease Diagnosis"

Our method is implemented similar to the Scikit-Learn Python API for easy usage on any experiments and is available in `enrich.py`. 

## Environment Setup

Code is written in Python 3.6+ and dependencies are managed by the Pipfile which can be conveniently used with `pipenv`. 

You can run the following to set up your environment and run an example my_file.py.

```bash
pipenv install 
pipenv shell
python my_file.py
```

## Usage

In your file, you can add:

```python 
from enrich import LongEnrichment

# Assuming n_subjects = number of subjects
# Assuming baselines is a list with each subject's baseline at the ith index
# Assuming longitudinal_data is a list with each subject's longitudinal data at the ith index

enriched_data = []
for subject in range(n_subjects):
    # Initialize Enricher 
    enricher = LongEnrichment(r=3, p=0.25)
    # Learn subject's patterns
    enricher.fit(longitudinal_data[subject])
    # Enrich the baseline data
    enriched_subject = enricher.transform(baselines[subject])
    # Store enriched representation for that subject
    enriched_data.append(enriched_subject)

    # Optionally, you can save the learned W for patient i by accessing enricher.W
```

## Contact

Feel free to create an issue on this repo if you have any questions. 

## Citing

TBD