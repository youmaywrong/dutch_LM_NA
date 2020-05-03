### Ablation Study of a Dutch LSTM LM using NA tasks

This repository contains all the code and data used to create the Dutch NA-tasks for my bachelor's project.
- `vocabulary`: all the words used to create the sentences
    - `vocab.txt` contains the 50,000 most common words in the corpus that the LSTM has been trained on
- `generated_data`: all possible sentences generated using the vocabulary and the grammars, capped off at 1000,000 and some restrictions for the maximum number of words per type
