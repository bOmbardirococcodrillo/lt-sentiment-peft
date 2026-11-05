# Error comparison: TF-IDF vs LoRA (validation set)

LoRA = `lora_r8_all_nfull`, a review counts as a LoRA error if it was wrong in at least 2 of 3 seeds.

- TF-IDF errors: 250
- LoRA errors (stable): 266
- both wrong: 164
- only TF-IDF wrong (LoRA fixed): 86
- only LoRA wrong (new LoRA errors): 102

## Properties of the reviews in each group

| Property | both wrong | only TF-IDF wrong (LoRA fixed it) | only LoRA wrong (LoRA broke it) | both correct |
|---|---|---|---|---|
| reviews | 164 | 86 | 102 | 3777 |
| % true label negative | 29.3 | 48.8 | 26.5 | 19.9 |
| median words | 14 | 11 | 10 | 7 |
| % very short (≤3 words) | 9.1 | 15.1 | 4.9 | 15.8 |
| % long (>40 words) | 12.8 | 7 | 4.9 | 6.8 |
| % no Lithuanian letters (5+ words) | 7.3 | 8.1 | 9.8 | 10.5 |
| % with negation | 64.6 | 51.2 | 58.8 | 30.8 |
| % from atsiliepimai.lt | 4.9 | 15.1 | 7.8 | 7.6 |
