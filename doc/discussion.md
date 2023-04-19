# Discussion

## Claim detection
**_Problem_**  
Media articles feature lots of compound claims ie containing several
statements in one sentence. 
It "confuses" the verification models that are trained on atomic claims.

**_Example_**
> tbd
> 

**_Potential solution_**  
Create a dataset of atomic scientifically verifiable claims 
and train a model to locate the bounds of such claims in the full text
rather that splitting the text first and then classifying a sentence as 
claim - not claim.
---
**_Problem_**  
Context is getting lost when the media article text is split into individual 
sentences

**_Example_**
> tbd
>

**_Potential solution_**  
Using co-reference resolution on the original media article text.
We tried doing it manually and came to a conclusion

## Predictions quality
- Heavily depends on the input claims quality discussed above
- The absense of proper dataset
- Fine-tuning MultiVerS model on Climate-FEVER

## Ideas to implement

1. It would be interesting to see the percentage of the articles from 
the full corpus that gets retrieved by the searches.
2. Create a curated dataset of claims extracted from media articles vs 
scientific articles abstracts by passing all the available 
[climate-news-db](https://www.climate-news-db.com/)  articles through the MultiVerS predictions.
3. Re-ranking before the match using the ideas and models 
from semanticscholar relevancy search described in this 
[blog article](https://blog.allenai.org/building-a-better-search-engine-for-semantic-scholar-ea23a0b661e7) 
and available in [github](https://github.com/allenai/s2search)
4. Verify non-scientific factual claims (eg environmental claims from firms and corporations). 
For that environmental claims
[dataset](https://huggingface.co/datasets/climatebert/environmental_claims) 
and [model](https://huggingface.co/climatebert/environmental-claims)
from ClimateBERT creators can be extremely useful