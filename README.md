## Creativity Assessment Poly-Encoder | YesNLP Lab, KSU

---

Hello developers and researchers! This Github repository is the extension of the academic paper on using poly-encoders (Humeau et al., 2019) for automated creativity asessment.

This is a full guide on implementation details, and how you can run the poly-encoder on your local machine or via SSH. 

---

## Poly-Encoder Details

### Architecture

<img src=poly%20encoder%20architecture.png  alt="alt-text" />

1. The poly-encoder first separately encodes contexts (question/prompt) and candidate (response). 
2. The poly encoder then attends *m* learnable codes to contexts, then attends candidates to this to generate a final
3. Poly-context and individual candidates are passed through a regression head to generate a single scalar score

See exact details in `model/polyencoder.py`

---

## Training

Populate train, val, and test files in `DataCleaning.py`
Log into WandB (in terminal, `>> wandb login`, change project name to desired name in `model/train.py`

### CPU (not recommended) / Local GPU
1. Simply run train.py

### GPU via SSH (recommended)
1. Transfer train/val/test files to your local instance, update paths in `model/train.py` as appropriate
2. Connect to instance via SSH
3. Clone updated repository
4. Create python venv
5. Change directory to `/model`
6. Run train.py

---

## Future modifications
* Researchers are encouraged to find better datasets to train on, and even apply poly-encoders to other multi-sentence scoring tasks
* Using other text encoders may positively change results
* All small model details in `model/polyencoder.py` can be easily tweaked
* We encourage you to contact us if you have any questions or discover anythign new! Email: sam.grouchnikov@gmail.com

---

Code By: Sam Grouchnikov\
Project By: Jiho Noh, Sam Grouchnikov, Philip Gregory