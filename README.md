### Ablation Study of a Dutch LSTM LM using NA tasks

This repository contains all the code and data used to create the Dutch NA-tasks for my bachelor's project and thesis.

- `data`: all NA tasks and the vocabulary used
    - `vocabulary`: all words used to create the sentences
        - `vocab.txt` contains the 50,000 most common words in the corpus that the LSTM has been trained on
    - `full_data`: all possible sentences generated using the words in `vocabulary`, with restrictions for some templates on the maximum number of possible words per word type (to avoid combinatorial explosion)
    - `tasks`: for each task and condition (singular/plural and combinations) a random sample from `generated_data`
- `output`: each .info file corresponds to a template and contains the accuracy for each condition
