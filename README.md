# Streamlit application for news articles Scientific verification and Global warming stance detection
This application and 
[the corresponding API](https://github.com/aaalexlit/cc-evidences-api/tree/main)
were created as an outcome of the 
[Detecting Bias in Climate Reporting in English and German Language News Media](https://omdena.com/projects/detecting-bias-in-climate-reporting-in-english-and-german-language-news-media/
) challenge by [Omdena](https://omdena.com/)

The ultimate goal of the challenge is to aid fighting misinformation 
in Climate Change-related articles across the internet.
The goal of the challenge is to create an AI-powered Bias detector for 
Climate Change related news articles.

This application attempts to detect two type of bias:
1. Scientific inaccuracy
2. Global Warming Stance

## General workflow

### Scientific verification

```mermaid
%%{ init: { 'theme': 'dark' } }%%
flowchart TB
   subgraph client1 [This Application]
      A(Media Article text or URL) -->|Split into sentences| S1("Sentence 1")
      A:::curAppNode -->|Split into sentences| S2("Sentence 2")
      A -->|Split into sentences| SN("...")
      S1:::curAppNode --> CR{"Climate related?\n (Optional)"}
      S2:::curAppNode --> CR
      SN:::curAppNode --> CR
      CR:::curAppNode -->|Yes| IC{"Is a claim? \n(Optional)"}
      CR:::curAppNode --No --x N[ Ignore ]
      IC:::curAppNode --No --x N1[Ignore]
   end
   subgraph API [Evidence Retrieval/Verification API]
      IC -- Yes ---> E["Retrieve Top k most similar evidences"]
      E --> R["Re-rank using citation metrics (Optional)"]
      R --> VC[["Validate with Climate-BERT based model"]]
   end
   subgraph client2 [ This Application ]
      R ---> VM[["Validate with MultiVerS"]]
      VC --> D["Display predictions"]
      VM:::curAppNode --> D:::curAppNode
   end
    style R stroke:#808080,stroke-width:2px,stroke-dasharray: 5 5
    style CR stroke:#808080,stroke-width:2px,stroke-dasharray: 5 5
    style IC stroke:#808080,stroke-width:2px,stroke-dasharray: 5 5
    classDef curAppNode fill:#F7BEC0,color:#C85250
    style client1 fill:#E9EAE0,color:#E7625F
    style client2 fill:#E9EAE0,color:#E7625F
    linkStyle 0,1,2,3,4,5,6,7,8,9,12,13,14 stroke:#F7BEC0,stroke-width:4px,color:#C85250
```

### Global Warming Stance detection
```mermaid
flowchart TB
    subgraph client1 [This Application]
        direction TB
        A(Media Article text) -->|Split into sentences| S1("Sentence 1")
         A:::curAppNode -->|Split into sentences| S2("Sentence 2")
         A -->|Split into sentences| SN("...")
         S1:::curAppNode --> CR{"Climate related?\n (Optional)"}
         S2:::curAppNode --> CR
         SN:::curAppNode --> CR
         CR:::curAppNode -- Yes --> S[Global Warming Stance Detection model]
         CR:::curAppNode -- No --x Ignore
         S:::curAppNode --> D[Display predictions]
         D:::curAppNode
       style CR stroke:#808080,stroke-width:2px,stroke-dasharray: 5 5
       style client1 fill:#E9EAE0,color:#E7625F
       classDef curAppNode fill:#F7BEC0,color:#C85250
    end
```

## Components

### Split into sentences
[Spacy "en_core_web_sm" pipeline](https://spacy.io/models/en#en_core_web_sm)
is used for text segmentation task  
This model is the smallest and the fastest and according to spacy's 
[Accuracy Evaluation](https://spacy.io/models/en#en_core_web_sm-accuracy) has
the same metric values as the bigger CPU-optimized models

### Classify as Climate-related (Optional)

## Discussion and next steps
Please refer to the [Discussion](doc/discussion.md)

## Local development and deployment
Please refer to the [Technical documentation](doc/tech.md)