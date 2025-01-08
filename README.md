# Embeddings for AIND metadata chatbot

The AIND metadata assets are a series of heavily nested JSON files. Due to the data's semi-structured nature, additional steps were required to prepare the data for chatbot processing. The database used is AWS' DocDB, hosted on MongoDB. The current embedding model is Amazon's `titan-embed-2.0`.

The embedding procedure for metadata aims to help optimize chatbot reponses to queries about specific assets and subjects. It performs better on semantic searches compared to quantifiable queries. As a result, the vector store picks up the slack when MongoDB aggregation pipelines don't suffice to find the appropriate response.

## Key steps to highlight:

### Modality types
Physiology (PHYSIO) and Microscopy (SPIM) modalities are encoded differently in the database. Specifically, SPIM experiments consist of instruments and acquisition files wheras PHYSIO experiments consist of rig and session files, respectively. Hence a general embedding algorithm cannot be applied to all assets. The two types of modalities had to be split and processed differently. 
 
### Recursive JSON splitter
Most frequently searched and nested fields in the metadata are data_description, subject, procedures, along with the 2 aforementioned fields for the PHYSIO and SPIM modalities. These fields are put through Langchain's `RecursiveJSONsplitter` to be chunked in preparation for vectorizing. The chunking process preserves the nesting structure of the field. So, no matter how nested a valueu is, the chunk will preserve it's parents and the hierarchy.

### Metadata for metadata
After vectorization, each embedding is stored in MongoDB along with the `subject_id`, `name` and `modality` fields. This allowd for the retrieval system of the chatbot to filter only relevant information. Furthermore, indexes have been applied to these fields in MongoDB, decreasing retrieval times by almost 100%.
