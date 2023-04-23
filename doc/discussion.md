# Discussion

## Claim detection
**_Problem_**  
Both media articles and scientific paper abstracts 
feature lots of compound claims ie containing several
statements in one sentence. 
It "confuses" the verification models that are trained on atomic claims.

***Example of a composite claim from scientific paper abstract***
> Our analysis suggests that the early twentieth century warming can best 
> be explained by a combination of warming due to increases 
> in greenhouse gases and natural forcing, some cooling due to 
> other anthropogenic forcings, and a substantial, but not implausible, 
> contribution from internal variability.
> 
That's predicted to support atomic claim `Natural variation explains a substantial 
part of global warming observed since 1850`

***Example of a composite claim from  media articles***
> While a warming globe might eventually be the dominant cause of Greenland’s 
> shrinking ice, natural cycles in temperatures and currents in the North Atlantic 
> that extend for decades have been a much more important influence since 1900.
> 

Which is predicted to be supported by `The Arctic featured the strongest surface 
warming over the globe during the recent decades, and the temperature increase 
was accompanied by a rapid decline in sea ice extent.`

***Potential solution***  
Create a dataset of atomic scientifically verifiable claims 
and train a model to locate the bounds of such claims in the full text
rather that splitting the text first and then classifying a sentence as 
claim - not claim.
---
***Problem***  
Context is getting lost when the media article text is split into individual 
sentences

***Example***
> tbd
>

***Potential solution***  
Using co-reference resolution on the original media article text 
as well as the indexed scientific paper abstracts

## Predictions quality and Further improvements

There are plenty of things to improve, current work just barely 
touched the surface.


- Heavily depends on the input claims quality as discussed above
- The absense of proper dataset. Climate-FEVER's source is Wikipedia 
which differs from both media articles and scientific abstracts hence
not exactly in the domain of out interest
- MultiVerS model that we're using is pretrained on out-of-domain data(FEVER)
and weakly-labeled partly-in-domain data (in a sense that in spite of being 
scientific articles, they are biomedical rather than environment-related) - 
PubMedQA + Evidence Inference.

### Fine-tune MultiVerS model on CLIMATE-FEVER dataset
For that CLIMATE-FEVER needs to be transformed into MultiVerS 
input format.

### Combine predictions of the 2 used models in different ways
1. Take as positive only those articles where both models 
agreed on the label 
2. Pass to the ClimateBERT-based model only phrases from articles
that were labeled by MultiVerS model as SUPPORTS or REFUTES

### Create in-domain (claims from news vs scientific abstracts) dataset

Potential approaches:

1. Using this tool and model's predictions as a base for expert review
2. Manually from extracting from [https://climatefeedback.org/](https://climatefeedback.org/)'s
reviews 

The advantage of the latter is that less expertise is required from the annotators
because the reviews are already made by climate specialists

## Other ideas to implement

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