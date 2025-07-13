# Cell Representation Learning

This project was created for the Computer Vision course at University Cologne. 

## Directory structure

`/experiments`: Code that does not belong to the pipeline but was used to
experiment with the dataset and different concepts 

`/plots`: Example images and plots of the dimensionality reduced embeddings 
with their labels or clustering algorithms applied

`/video_lists`: .json files containing different video lists used in training

## Used Technologies
- Python 3.11.5
- torch (cuda 12.6)
- requirements from `requirements.txt` (install with `pip install -r requirements.txt`)

## Contributors
Catharina Dümmen, 
Florian Bremm

## License
For academic use only. Not licensed for commercial reuse.

## Execute Pipeline
Execute the preprocessing.py Notebook

`python training.py {training_name} {gpu number} {path to video list}`

`python clustering.py`