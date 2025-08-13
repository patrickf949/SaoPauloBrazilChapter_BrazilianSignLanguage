---
title: Report Page
layout: post
permalink: /report
author: Ben Thompson
---

# **Introduction**

The problem statement, goals, and task outline we kicked the project off with can be found at [https://www.omdena.com/chapter-challenges/ai-assisted-sign-language-translation-for-brazilian-healthcare-settings](https://web.archive.org/web/20250403045320/https://www.omdena.com/chapter-challenges/ai-assisted-sign-language-translation-for-brazilian-healthcare-settings).

The key technical points are:
> ***1. Accuracy:***
> - *Achieve at least 90% accuracy in recognizing and translating common Libras signs to Portuguese text*
> - *Reach 80% accuracy for medical-specific Libras terminology*

&
> ***3. Vocabulary Coverage:***
> - *Include at least 5,000 common Libras signs in the initial model*
> - *Incorporate a specialized medical vocabulary of at least 1,000 terms*


The scale of the project as written there and the other 4 points is very large, and very ambitious for a 3-month project. It is worth noting we did not begin the project with a high quality / quantity dataset. 

Considering that we didn't begin the project with a dataset, items in the 'Vocabulary Coverage' section were simply not feasible. So we knew we would have to focus on a subset of the problem statement and redefine the scope.

However we didn't abandon this problem statement, instead we used it's ambition to push ourselves to aim high, and it's vision as the basis for our decision making about the actual scope of the project. 

For example, when deciding which words in the dataset to focus on, we intentionally selected words that would be more likely to be used in a medical context over others. We also worked with the perspective that this is an initial proof-of-concept, of a solution we would like to develop further in future.


#  **Research & Planning**
To decide on the scope of the project and our plan for development, the first task was to conduct research and share findings with each other. 

We investigated:

- **Domain**: Sign Language Processing (SLP)
    - What are the conventions for processing Sign Language  data?
    - What are the common modelling approaches?
    - What are the pros and cons of each approach?
- **Data Sources**: LIBRAS Datasets
    - What public LIBRAS datasets are available?
    - What are the different formats of Sign Language video data?
- **Existing Literature**: Sign Language Processing with LIBRAS Data
    - Papers & projects exploring SLP for LIBRAS specifically
    - What datasets did they use?
    - What were there results, and can we replicate or improve on them?
    
## **Domain: Sign Language Processing (SLP)**

Sign Language Processing (SLP) is a field of artificial intelligence that combines Natural Language Processing (NLP) and Computer Vision (CV) to automatically process and analyze sign language content. Unlike spoken languages that use audio signals, signed languages use visual-gestural modality through manual articulations combined with non-manual elements like facial expressions and body movements.

### **Key SLP Tasks Relevant to Our Project:**

1. **Sign Language Recognition**: Identifying individual signs or sequences of signs from video input
2. **Sign Language Translation**: Converting sign language videos into spoken language text (our primary focus)
3. **Sign Language Production**: Generating sign language videos from spoken language text
4. **Sign Language Detection**: Determining if a video contains sign language content
5. **Sign Language Segmentation**: Identifying boundaries between different signs in continuous signing

### **Challenges in SLP:**
- **Visual-gestural modality**: Unlike spoken languages, signed languages lack a written form, forcing researchers to work directly with raw video signals
- **Simultaneity**: Multiple articulators (hands, face, body) can convey information simultaneously
- **Spatial coherence**: Spatial relationships and movements are crucial for meaning
- **Lack of standardization**: Limited annotated datasets, especially for languages like Brazilian Sign Language (Libras)

### **Common Approaches to SLP:**

#### **1. Data Representation:**
- **Video-based**: Process raw video directly (most common since sign languages have no written form)
- **Pose estimation**: Extract hand, face, and body keypoints using tools like MediaPipe or OpenPose
- **Symbolic notation**: Use intermediate representations like glosses (word-level labels)

#### **2. Feature Extraction:**
- **CNNs**: Extract visual features from video frames using deep learning models
- **Pose landmarks**: Use hand shapes, movements, and facial expressions as input features
- **Motion analysis**: Track movement patterns between video frames

#### **3. Sequence Modeling:**
- **LSTMs/RNNs**: Handle the time-dependent nature of sign language sequences
- **Transformers**: Use attention mechanisms for understanding sign relationships
- **Graph networks**: Model connections between different body parts

### **Recommended Reads:** 
- [Sign Language Processing - Overview of the Field](https://research.sign.mt/)
    - A comprehensive and up-to-date resource covering the field of SLP research and the dataset landscape
- [A review of deep learning-based approaches to sign language processing](https://www.tandfonline.com/doi/full/10.1080/01691864.2024.2442721)
    - A recent paper from covering the current state of the art in SLP, data availability, and the challenges of the field.


## **Data Sources: LIBRAS Datasets**

### **Review of available datasets**
We surveyed existing LIBRAS datasets, and reviewed their contents. This table contains details from the 9 most relevant datasets we found.

<table style="font-size: smaller; text-align: center; table-layout: fixed; width: 100%;">
<style>
table th, table td {
  padding-left: 6px;
  padding-right: 6px;
}
</style>
<colgroup>
<col style="width: 25%;">
<col style="width: 6%;">
<col style="width: 12%;">
<col style="width: 12%;">
<col style="width: 14%;">
<col style="width: 10%;">
<col style="width: 10%;">
<col style="width: 10%;">
</colgroup>
<tr>
<th style="text-align: left;">Dataset</th>
<th>Year</th>
<th>Type</th>
<th>Format</th>
<th>Accessibility</th>
<th>Number of Classes</th>
<th>Examples per Class</th>
<th>Total Examples</th>
</tr>
<tr>
<td style="text-align: left;">Brazilian Sign Language Words Recognition Dataset</td>
<td>-</td>
<td>Words</td>
<td>Images</td>
<td><a href="https://www.kaggle.com/datasets/alvarole/brazilian-sign-language-words-recognition">Downloadable</a></td>
<td>15</td>
<td>60 - 1000</td>
<td>~5000</td>
</tr>
<tr>
<td style="text-align: left;">Brazilian Sign Language Alphabet Dataset</td>
<td>2020</td>
<td>Alphabet</td>
<td>Images</td>
<td><a href="https://biankatpas.github.io/Brazilian-Sign-Language-Alphabet-Dataset/?utm_source=chatgpt.com">Downloadable</a></td>
<td>15</td>
<td>150 - 600</td>
<td>3000</td>
</tr>
<tr>
<td style="text-align: left;">Libras Movement</td>
<td>2009</td>
<td>'Hand Movement Types'</td>
<td>Hand Landmarks</td>
<td><a href="https://archive.ics.uci.edu/dataset/181/libras+movement">Downloadable</a></td>
<td>15</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td style="text-align: left;">Libras SignWriting Handshape (LSWH100)</td>
<td>2024</td>
<td>'Handshapes'</td>
<td>CG Images</td>
<td><a href="https://www.sciencedirect.com/science/article/pii/S2352340924007455">Downloadable</a></td>
<td>100</td>
<td>~480</td>
<td>48000</td>
</tr>
<tr>
<td style="text-align: left;">LIBRAS Cross-Dataset Study</td>
<td>2023</td>
<td>Words</td>
<td>Videos</td>
<td><a href="https://github.com/avellar-amanda/LIBRAS-Translation?tab=readme-ov-file">Restricted</a></td>
<td>49</td>
<td>~6</td>
<td>~294</td>
</tr>
<tr>
<td style="text-align: left;"><strong>V-Librasil - A New Dataset with Signs in Brazilian Sign Language (Libras)</strong></td>
<td>-</td>
<td>Words</td>
<td>Videos</td>
<td><a href="https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras">Scrapable</a></td>
<td>1,364</td>
<td>3</td>
<td>4089</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Federal University of Viçosa (UFV) - LIBRAS-Portuguese Dictionary Dataset</strong></td>
<td>-</td>
<td>Words</td>
<td>Videos</td>
<td><a href="https://sistemas.cead.ufv.br/capes/dicionario/">Scrapable</a></td>
<td>1,004</td>
<td>1 - 2</td>
<td>1029</td>
</tr>
<tr>
<td style="text-align: left;"><strong>National Institute of Deaf Education (INES) - LIBRAS Dictionary Dataset</strong></td>
<td>-</td>
<td>Words</td>
<td>Videos</td>
<td><a href="https://www.ines.gov.br/dicionario-de-libras/">Scrapable</a></td>
<td>237</td>
<td>1 - 2</td>
<td>282</td>
</tr>
<tr>
<td style="text-align: left;"><strong>SignBank - LIBRAS Dataset (Universidade Federal de Santa Catarina)</strong></td>
<td>-</td>
<td>Words</td>
<td>Videos</td>
<td><a href="https://signbank.libras.ufsc.br/en">Scrapable</a></td>
<td>3090</td>
<td>1</td>
<td>3090</td>
</tr>
</table>


<!-- | Dataset | Year | Type | Format | Access | Number of Classes | Examples per Class | Total Examples |
|---------|------|---------------|---------|---------|----------------|-------------|-------------|
| Brazilian Sign Language Words Recognition Dataset | - | Words | Images | [Downloadable](https://www.kaggle.com/datasets/alvarole/brazilian-sign-language-words-recognition) | 15 | ~60 - 1000 | ~5000 |
| Brazilian Sign Language Alphabet Dataset | 2020 | Alphabet | Images | [Downloadable](https://biankatpas.github.io/Brazilian-Sign-Language-Alphabet-Dataset/?utm_source=chatgpt.com) | 15 | 150 - 600 | 3000 |
| Libras Movement | 2009 | 'Hand Movement Types' | Hand Landmarks | [Downloadable](https://archive.ics.uci.edu/dataset/181/libras+movement) | 15 | - | - |
| Libras SignWriting Handshape (LSWH100) | 2024 | 'Handshapes' | CG Images | [Downloadable](https://www.sciencedirect.com/science/article/pii/S2352340924007455) | 100 | ~480 | 48000 |
| LIBRAS Cross-Dataset Study | 2023 | Words | Videos | [Restricted](https://github.com/avellar-amanda/LIBRAS-Translation?tab=readme-ov-file) | 49 | ~6 | ~294 |
| **V-Librasil - A New Dataset with Signs in Brazilian Sign Language (Libras)** | - | Words | Videos | [Scrapable](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras) | 1,364 | 3 | 4089 |
| **Federal University of Viçosa (UFV) – LIBRAS-Portuguese Dictionary Dataset** | - | Words | Videos | [Scrapable](https://sistemas.cead.ufv.br/capes/dicionario/) | 1,004 | 1 - 2 | 1029 |
| **National Institute of Deaf Education (INES) – LIBRAS Dictionary Dataset** | - | Words | Videos | [Scrapable](https://www.ines.gov.br/dicionario-de-libras/) | 237 | 1 - 2 | 282 |
| **SignBank - LIBRAS Dataset (Universidade Federal de Santa Catarina)** | - | Words | Videos | [Scrapable](https://signbank.libras.ufsc.br/en) | 3090 | 1 | 3090 | -->


### **Selecting datasets for our task**

We decided that although it is more difficult, we wanted to focus on sign **videos**, not images. 
 - This would be more applicable to a real-world situation.

We also decided to focus on sign **words**, not alphabet signs, movement/shape types, or sentences. 
 -  Again, **words** would be more applicable to a real-world situation than alphabet signs or movement/shape types. 

Sign sentences would be the most similar to real-world use, and also the most challenging to model accurately. 
  - However we didn't find any datasets with this type of data. 
  - Even in SLP more broadly, large high quality datasets of sentences are rare, so it is not surprising we couldn't find any for LIBRAS.

With these criteria in mind, the datasets we selected for our task were:
- [INES](https://www.ines.gov.br/dicionario-de-libras/)
    - National Institute of Deaf Education (INES) – LIBRAS Dictionary Dataset
- [Signbank](https://signbank.libras.ufsc.br/pt)
    - SignBank - LIBRAS Dataset (Universidade Federal de Santa Catarina)
- [UFV](https://sistemas.cead.ufv.br/capes/dicionario/)
    - Federal University of Viçosa (UFV) – LIBRAS-Portuguese Dictionary Dataset
- [V-Librasil](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras#files)
    - V-Librasil - A New Dataset with Signs in Brazilian Sign Language (Libras)

*(All 4 highlighted in bold in the table above)*

## **Exisiting Literature: Sign Language Processing with LIBRAS Data**

We found a few papers working with LIBRAS data, but most were not directly relevant to our project, as they focussed on different tasks. Like images instead of videos, or classification of movement types instead of words.

### **2023 Cross-Dataset Study**

The most relevant paper we found was the 2023 paper [A Cross-Dataset Study on the Brazilian Sign Language Translation](https://openaccess.thecvf.com/content/ICCV2023W/CLVL/papers/de_Avellar_Sarmento_A_Cross-Dataset_Study_on_the_Brazilian_Sign_Language_Translation_ICCVW_2023_paper.pdf) by de Avellar & Sarmento.

Their work was a cross-dataset study on the Brazilian Sign Language Translation task, and it was very relevant to the goals of our project. The information in the paper was a good reference point for us to confirm our plan and approach was sound. 

#### **Dataset**

They combined the same 4 datasets we selected for our task, and made a lot of effort to scrape the data, clean, and preprocess it into one unified dataset. This is the 2023 'LIBRAS Cross-Dataset Study' dataset listed in the table above. Kindly the dataset is available on request, but unfortunately we didn't hear back from one of the authors. So we would have to scrape, clean, and preprocess the data ourselves.

We investigated if any new datasets had become available since the paper's publishing, or if any of the existing datasets had been expanded. But it was not the case, so we would be using the same source datasets. We planned to differentiate our approach from theirs with more detailed preprocessing to address the data source variation. Also, our dataset selection was more focused on healthcare related words, in line with the original goal, adding another point of difference.

#### **Modelling**

After filtering the dataset down to 49 words, with an average of 6 videos per word, they used an LSTM model to classify the signs. They tried two different methods for feature extraction: pre-trained Convolutional Neural Network (CNN) models, and Landmark Estimation. Based on our review of wider SLP literature, we concluded that the modelling approach they followed was still appropriate for us.  We planned to deviate from their work by taking a different approach to data augmentation and spending time engineering new features.

#### **Results**

The results were better with the LSTM model than the CNN model.

On the full 49 word dataset, the best model achieved:
 - Test accuracy of 41%
 - Test top-5 accuracy of 75%

Looking at the commonly misclassified words, they saw that the signs had a lot of variation in the way they were signed. Removing these to have a lower variance dataset of 33 words, the best model achieved:
 - Test accuracy of 66%
 - Test top-5 accuracy of 94%

## **Plan**

### **Overview**
We will develop AI models to classify videos of LIBRAS word signs
- Not images of signs, and not alphabet signs

If possible, we will focus on healthcare related words, in line with the original goal
- But since we have to be realistic with the data available to us, and this is just a PoC, the scope is flexible
- If the data is limited in quantity or quality, we will focus on general words

We will create our own dataset by scraping videos from 4 public data sources.

We will extract pose landmarks with an open source estimation model, engineer additional features, and try using RNN, LSTM & Transformer model architectures to classify the signs.

The deliverables will be a web hosted report and a demo application.

### **Tasks**

The project work will be divided into key tasks, which are each managed by Task Leads.

#### **Data Collection** [Task Leader: Ayushya Pare]
- Develop code to scrape videos and metadata from the following 4 data sources:
    - [INES](https://www.ines.gov.br/dicionario-de-libras/)
    - [Signbank (UFSC)](https://signbank.libras.ufsc.br/pt)
    - [UFV Dictionary](https://sistemas.cead.ufv.br/capes/dicionario/)
    - [V-Librasil Dataset](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras#files)

#### **Data Cleaning & Organization** [Task Leader: Gustavo de Paula Santos]
- Clean the metadata from each data source, and process them into a unified format for easy management and analysis of the dataset.
- Define some criteria to narrow down the dataset, manually review the subset of potentially usable videos
- Decide the minimum number of videos per word based on the available data, and finalize our project dataset.

#### **Preprocessing & Data Augmentation** [Task Leader: Ben Thompson]
- Implement an open source pose estimation model to extract hand, face, and body landmark keypoints from the videos
- Develop a preprocessing pipeline for the landmark data for each video, retaining valuable information about the signer's position and movement, but reducing non-informative variation between data sources
- Design appropriate data augmentation techniques to mitigate issues with having such a small number of examples per class

#### **Landmark Features -> Model** [Task Leader: Anastasiia Derzhanskaia]
- Engineer a variety of informative features from the landmark data
- Develop a robust model training pipeline so that many members can contribute to running and evaluating experiments
- Explore a variety of model architectures for the task: RNN, LSTM, and Transformer

#### **Demo Application Development** [Task Leader: Patrick Fitz]
- Develop and deploy a demo application that uses the final trained model to classify LIBRAS videos
- The user can select from our library of words, or upload their own video


# **Data Collection**


## **Scraping**

To build a robust dataset of BSL videos, we developed code to scrape videos and metadata from the following 4 data sources:
* [INES](https://www.ines.gov.br/dicionario-de-libras/)
* [Signbank (UFSC)](https://signbank.libras.ufsc.br/pt)
* [UFV Dictionary](https://sistemas.cead.ufv.br/capes/dicionario/)
* [V-Librasil Dataset](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras#files)

### **Web Scraping Automation Approach:**

Due to the scale and structure of these 4 data sources, we implemented web automation tools—primarily **Selenium**—to efficiently extract video URLs and relevant metadata. Each website had different structures, and the patterns were often unclear or inconsistent. So the task required carefully considered scraping code development for each data source. 

For efficient processing, we initally just **scraped metadata and download URLs** for each video, rather than directly downloading thousands of video files that we might not use.  Then, later in the project, after filtering the dataset down only to words we will potentially use in our target dataset, we can use the URLs to download only the videos we actually need.


### **Extracted Metadata:**

#### **Common Fields**

The available metadata varied slightly by source, but there were 4 common fields we made sure to include:
- `label` - The label associated with the video. Could be a letter, word, or phrase depending on the source.
- `video_url` - URL to the downloadable video file
- `signer_number` - Identifier for the person performing the sign in the video.
  - Sometimes taken directly from the source e.g. V-Librasil
  - Sometimes assigned by us e.g. SignBank
  - Left as 0 when it hasn't been reviewed
- `data_source` - Lowercase, two character string, indicating which data-source the entry belongs to.
  - `in` for INES
  - `sb` for SignBank
  - `uf` for UFV
  - `vl` for V-librasil

#### **Additional Useful Fields**

When a data source had additional metadata that was relevant to our task, we made sure to include it. Some examples of useful information:

INES had various lingusitic information about the sign that would be helpful during the review process to confirm what the sign referred to when the label was a homograph.
- `assuntos` - Subject/topic categories
- `acepção` - Definition/meaning of the word
- `exemplo` - Example sentence in Portuguese
- `classe gramatical` - Grammatical class

SignBank sometimes had multiple videos for the same word, indicating additional videos with numbering. We observed this usually meant there were multiple sign variants for the same word, and they had recorded a video for each variant.

This would be important information to have when finalising our dataset, as we should try to include one sign variant per word, and make sure each data source is using the same variant. 
- so we collected metadata about the `sign_variant`, by processing the `label` column
- e.g. 'FAMÍLIA' -> `1` & 'FAMÍLIA-2' -> `2`
- They might be different sign variants for the same word


# **Data Cleaning & Organization**

With the metadata for each data source collected, we spent time cleaning the data, and unifying it into a single file for easy management and analysis of the dataset.

We will then clean the metadata from each data source, and process them into a unified format for easy management and analysis of the dataset.

We will define some criteria to narrow down the dataset, manually review the subset of potentially usable videos, and finalize our project dataset.

## **Cleaning the Dataset**

A lot of cleaning was done for each individual data source's `metadata.csv`, but some cleaning was also needed for the combined metadata:
- Unifying labels
- Removing duplicates
- Removing videos with broken URLs

### **Unifying labels**

Cleaning and unifying the formatting of the labels would allow us to compare videos with the same label across data sources.

- **INES**
  - `FAMÍLIA` -> `família`
- **SignBank**
  - `FAMÍLIA` -> `família`
  - `FAMÍLIA-2` -> `família`
- **UFV**
  - `Família` -> `família`
- **V-Librasil**
  - `Família` -> `família`


### **Reviewing the Dataset**

Combining the 4 data sources, we had information for 8,490 videos. 

To clean the dataset further, we would need to go past the metadata, and review some videos in the dataset directly.

#### **Homographs:**

Like any other language, there are some cases of homographs in Portuguese, that is, words that have the same spelling and different meanings. An example of homograph in English is the word "bat" that could either refer to the animal or the object used to hit the ball in a baseball game. 

Due to the structure of the data, these words would be registered with the same label but have different signs. So we had to review the videos, and determine which meaning the label referred to.

We used the fact that V-Librasil does not have any possible intra-dataset homographs to our advantage- each label in the dataset has one set of sign videos. Because both SignBank and INES datasets have multiple signs corresponding to a label with the same spelling in Portuguese, it would be quite difficult to choose which signs to use based solely on the meaning of the words. 

These are the steps taken to choose the version of the signs:
1. Use V-Librasil as baseline dataset.
2. Compare videos from both SB and NE to find the best match in the signs
3. For every match, count as a possible word

#### **Label Synonyms:** 

Looking for ways to increase the number of videos for each word in our target dataset, we considered synonyms. Sign language signs do not have a 1-1 correspondence to words in spoken languages. For example, the same sign could be used for the word `big` and the word `large`. In this case, the same sign could be labelled as `big` in one data source, and `large` in another. Identifying these would allow us to find additional sign videos for a certain label.

Reviewing this manually would have been very time consuming. We considered developing some approach to use LLM Tokenizer embeddings to find labels that were closest in meaning to narrow down the possible synonyms, and then potentially even a feature extractor on the videos to see if the signs were also similar.

However since the potential increase in data for each word would be minimal, we decided to narrow down our target dataset first, assuming no synonyms, then review any synonyms for only those cases. In the end we didn't find any for our target dataset.

#### **Sign Variants:**

Some words could have more than one way to sign them, depending on the region or signer. Similarly to the homographs, these words would be registred under the same label but have different signs in the videos. Except in the case of SignBank, where they indicate multiple sign variants for the same word.

For this we manually reviewed videos for some signs, and recorded which sign variant should be used for the label. This would be quite time consuming, so we did this after narrowing down the dataset to the candidates for the target dataset.

Steps:
1. Use SignBank as the reference dataset
2. For all labels that have multiple sign variants in SignBank, review the videos from all other data sources 
3. Determine if the other data sources all use the same sign variant for the label
4. Determine which video in SignBank matches that sign variant

## **Organizing the Dataset**

With our cleaned & reviewed dataset, we could start narrowing down the dataset to the words we will use in our target dataset.

#### **Number of videos per word**

Even though there were videos for every registered word in the datasets, some of them had fewer videos than others. For the project to be successful, it was necessary to have at least a certain amount of example videos to train the models.

We also cared about having a variety of data sources for each word. V-Librasil always had 3 videos per word, but we didn't want to rely on just one data source, so if a word wasn't also present in the other data sources, the variety of features in the videos would be limited.

So first we narrowed down the dataset with 2 criteria:
- Words that appear in at least 3 data sources
- Words that have at least 5 videos

This gave us 170 words which were candidates for our target dataset.

#### **Healthcare related words:** 

The project's main goal was to translate LIBRAS specifically in healthcare settings. Therefore, it was necessary to select health related words to compose the target dataset.

Focusing on food, body parts, medical things, and other common words, the candidates were narrowed down to 46 words.


### **Final dataset**

Among the words in these 46 candidates, the majority had 6 videos, so we made our criteria a bit stricter, first removing the words that had less than 6 videos.

Then with this shorter list, we spent time reviewing the videos, to ensure the signs were all the same variant. Removing words with less than 6 videos of the same sign, we confirmed our final target dataset.

Our final dataset consisted of 25 words, and a total of 150 videos.

Each word had 6 videos, from the 4 data sources.


### **Table of the words in the final dataset**

|---|---|---|---|---|
| **Brazilian** | Ajudar | Animal | Aniversário | Ano | Banana |
| ***English***   | *Help*   | *Animal* | *Birthday*    | *Year* | *Banana* |
| **Brazilian** | Banheiro | Bebê | Cabeça | Café | Carne |
| ***English***   | *Bathroom* | *Baby* | *Head* | *Coffee* | *Meat* |
| **Brazilian** | Casa | Cebola | Comer | Cortar | Crescer |
| ***English***   | *House* | *Onion* | *Eat* | *Cut* | *Grow* |
| **Brazilian** | Família | Filho | Garganta | Homem | Jovem |
| ***English***   | *Family* | *Son* | *Throat* | *Man* | *Young* |
| **Brazilian** | Ouvir | Pai | Sopa | Sorvete | Vagina |
| ***English***   | *Hear* | *Father* | *Soup* | *Ice Cream* | *Vagina* |

|---|---|---|---|---|
| **Data Source** | INES | SignBank | UFV | V-Librasil |
| **Number of Videos**   | 1 | 1 | 1 | 3 |

<div align="center">
<img src="assets/all_signs_for_banana.gif" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p><em>All 6 videos in the dataset for the word 'banana'</em></p>
</div>


# **Data Preprocessing**


## **Exploring the videos in the dataset**

There are significant differences in the videos between the data sources, and also some differences within data sources.

### **Frame Rate**
Looking at the frame rate of the videos, we can see that the data sources have a wide range of frame rates.

<div align="center">
<img src="assets/fps_bar_chart.png" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em>Stacked bar chart showing the distribution of frame rates, categorized by data source</em></p>
</div>


Across the full dataset, the majority of videos have a frame rate of 60 fps.

For most data sources, all videos have the same frame rate. Except for V-Librasil, where we have examples with 24, 30, and 60 fps.

### **Video Dimensions**

Looking at the dimensions of the videos, we can see that the data sources also have a wide range of dimensions.

<div align="center">
<img src="assets/video_dimensions_for_animal.png" alt="isolated" title="hover hint" style="width: 95%; border: 2px solid #ddd;"/>
<p><em>Visualisation of the various video dimensions for the word 'animal'</em></p>
</div>

The range of dimensions is quite large, from 240x176 to 1920x1080. So we will need to take care to standardise the dimensions of the data, without losing information.

<div align="center">
<img src="assets/video_dimensions_bar_chart.png" alt="isolated" title="hover hint" style="width: 95%; border: 2px solid #ddd;"/>
<p><em>Stacked bar chart showing the distribution of video dimensions, categorized by data source</em></p>
</div>

Across the full dataset, the majority of videos are 1920x1080p.

For most data sources, videos can have two different dimensions. Except for INES, where all examples are 240x176

### **Video Durations**

<div align="center">
<img src="assets/video_durations_orig_boxplot.png" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em>Boxplot of the video durations for the unprocessed dataset, categorized by data source</em></p>
</div>

Looking at the distribution of video durations for each data source, we can see that there is quite a difference between the data sources. 

It is expected that there will be variation within data sources, because some signs are shorter than others, some longer. On this point, the range of durations is somwhat similar between INES, SignBank, and UFV. V-Librasil has a much wider range of durations.

Inspecting the videos, we can quickly see what this plot is representing. The signing speed for INES is clearly much faster than the other data sources. But INES videos also usually have less time where the signer is paused at the beginning and end of the video compared to the other data sources. 

We can also see that the V-Librasil signing speed is much slower than the other data sources, even across different signers. But this also seems to be due to the video speed. The videos appear to be in slow motion. to some degree.

We will apply some preprocessing to the videos to remove the pauses at the beginning and end of the video, since they don't contain any information about the sign. We will also sample frames from the videos as part of the data augmentation process, mitigating the large difference in speed between the data sources.

## **Summary of the Preprocessing Pipeline**

Our preprocessing pipeline transforms raw video data into standardized landmark sequences suitable for machine learning. The pipeline consists of four main steps:

1. **Pose Estimation**: Extract 543 landmarks (pose, face, hands) from each video frame using MediaPipe Holistic
2. **Motion Detection & Trimming**: Measure motion between frames, use thresholds to detect sign start/end points, then trim videos to include only the actual signing performance
3. **Scaling & Alignment**: Normalize signer scale and position across all data sources while preserving relative motion within each video
4. **Interpolation**: Fill missing landmarks (`None` values) using forward/backward fill for start/end frames and linear interpolation for middle frames


<div align="center">
<img src="assets/pose_for_casa_on_black_preproc.gif" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p><em>GIF showing pose estimation landmarks before and after preprocessing for the sign 'casa'</em></p>
</div>

## **Pose estimation with MediaPipe Holistic**

The first preprocessing step was pose estimation on the videos. No preprocessing was done on the videos themselves. The resulting pose landmarks would be used:
- In preprocessing for motion detection, offsetting, and scaling
- As the base features that will be input to the model

We used the [MediaPipe Holistic](https://ai.google.dev/edge/mediapipe/solutions/guide) model to estimate pose landmarks for each frame.

<div align="center">
<img src="assets/pose_for_casa_on_orig_and_black_raw.gif" alt="isolated" title="hover hint" style="width: 95%; border: 2px solid #ddd;"/>
<p><em>GIF showing raw pose estimation landmarks for the sign 'casa'</em></p>
</div>

### **MediaPipe Holistic Features:**
- **Open Source:** Freely available under Apache 2.0 license for development and modification
- **Multi-Model Integration:** Combines pose estimation, face landmark detection, and hand tracking into a unified pipeline.
- **Comprehensive Detection:** Detects 543 total landmarks (33 pose, 468 face, 21 per hand) for full-body analysis. Returns landmarks for each frame, and uses the information across all frames to improve the accuracy for each individual frame.
- **Landmark Precision:** Each landmark includes x, y, z coordinates with confidence scores for reliability assessment. Low confidence landmarks will be returned as `None`, so quality can be assured.
- **Structured Output:** Provides landmark coordinates in a standardized format for consistent data processing, with coordinates normalized between 0.0 and 1.0 relative to image dimensions. Ensuring videos with different resolutions produce results in the same format.

<!-- 
We didn't do any preprocessing on the videos themselves. We wanted our classification pipeline to be compatible with a wide range of videos, and most open source pose estimation models are flexible to a wide range of outputs, while giving a standard output. We also thought that Pose Landmark information would be more easy to work with during preprocessing, giving us more flexibility to unify the features of the data sources. -->


## **Start/End Point Trimming**

The next preprocessing step was to trim each video to include only the actual sign performance, not the pause before and after. To do this, we developed a method to automatically detect the start and end points of signing based on motion. This allowed us to remove the periods at the beginning and end where the signer is stationary, resulting in shorter clips focused on the sign itself.

### **Motion Detection**

#### **Exploring Motion Detection Methods**

We explored various different methods for measuring motion between frames. You can see the results of the three main measurement methods we used for the word 'aniversário' from the INES data source below.

<div align="center">
<h3><b><i>Motion Detection for "Aniversário" (INES data source)</i></b></h3>
<img src="assets/motion_for_aniversario_all_ma.gif" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em></em></p>
</div>

**1. Absolute Frame Difference**
- Simple and computationally efficient
- Compares consecutive frames using cv2.absdiff() to find pixel-wise differences
- Sensitive to camera shake & lighting changes, and can detect noise as motion

**2. Background Subtraction**
- Uses OpenCV's MOG2 (Mixture of Gaussians) background subtractor to build a statistical model of the background over time
- Identifies foreground objects by comparing current frame against learned background over time
- Measures motion intensity by counting non-zero pixels in the foreground mask

**3. Landmarks Difference**
- Analyzes Euclidean distance changes between MediaPipe landmark positions across consecutive frames
- Supports pose, face, and hand landmarks with configurable inclusion/exclusion
- Combines landmark distances using methods: mean, median, max, or RMS (root mean square)

Exploring each method individually, we found that while they all had slight differences in the type of motion they were best at detecting, they were all generally good at detecting the peaks in motion at the beginning and end of the sign. We also explored using weighted combinations of all methods, to try to have a more robust method by having the best of each.


#### **Final Motion Detection Method**

In the end, we settled on using the **landmarks difference method only**. It was the most robust and consistent across data sources. We also preferred the simplicity of using one method over trying to find the best combination of multiple.

<div align="center">
<h3><b><i>Motion Detection for "Aniversário" (INES data source)</i></b></h3>
<img src="assets/motion_for_aniversario_lm.gif" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em></em></p>
</div>

We **used the `mean` combination method**, taking a simple average of all the frame-to-frame landmark distances for each frame to measure the motion. The other options were `median` (robust to outliers), `max` (considers only the largest movement) and `root mean square` (emphasizes larger movements).

We **excluded the face landmarks** from the computation, since those are 468 landmarks that generally don't move much in our dataset- the signer is standing still, with only slight head movements that don't always align with the start or end of the sign. For almost all of the combination methods (except `max`), the small distances for the 468 face landmarks would dominate the results over the 33 pose landmarks and 42 hand landmarks.

Since we cared more about identifying peaks in motion, rather than measuring the absolute value of the motion, we **normalised the motion measurements** for each individual series between 0 and 1. 

Since we had such a variety of frame rates, we also **applied a moving average to the motion measurements**, to smooth out the series and make the results more consistent between data sources. The window duration was chosen to be 0.334 seconds, which would be converted into the actual window size in frames based on the frame rate of the video.



### **Analysing Motion Start and End**

Now we had a series of motion measurements for each frame, we had to develop methods that use them to identify the start and end points of the sign. 

We explored various methods, but in the end we settled on a basic approach using thresholds to detect the start and end points of the sign:
- Set an motion threshold for the start & end
  - We found 0.2 for both was quite robust to the differences in each data source.
- From the beginning of the series, find the first frame where the motion crosses the threshold, return the previous frame as the start point
- From the end of the series, find the first frame where the motion crosses the threshold, return the next frame as the end point


<!-- We developed two different methods:

1. **Simple Method:**
- Set an motion threshold for the start & end
- From the beginning of the series, find the first frame where the motion crosses the threshold, return the previous frame as the start point
- From the end of the series, find the first frame where the motion crosses the threshold, return the next frame as the end point

2. **Complex Method:**
- Detects the first and last peak in motion data to get the general location of significant motion
- Calculates slopes (rate of change) of the motion data using a sliding window approach
- Finds peaks in the absolute slope values to identify inflection points where motion changes direction
- Searches backward from the first motion peak to find the nearest slope peak before it
- Searches forward from the last motion peak to find the nearest slope peak after it
- Applies configurable time buffers (default 0.15 seconds) before/after the detected slope peaks
- Falls back to the simple threshold method if no motion peaks or slope peaks are found 

In the end, we settled on using the **simple method** with start & end motion thresholds of 0.2. 
- Both methods have threshold parameters that need tuned
  - The simple method has only 2, and they are more intuitive to tune
  - The complex method has 6, and they are more difficult to tune -->


<!-- - in the final version we just went for the simple version 
    - both have threshold params that need tuned 
    - But simple has less and is more intuitive 
    - Without annotating our ground truth desired start / end, it is hard to tune this
        - all we could do was manually experiment 
- our method for tuning was just
    - run manual experiments for a bunch of thresholds
    - compare the variance of the start/end %
        - assumption is it should be less if there are less extreme cases
    - inspect the results of a sample of the data wit visualisation like these:
- from our limited time experimenting, complex was not significantly better than simple. 
    - with more time / annotations we could tune complex to be significantly better
    - Simple was also robust to jittery motion
    - We would need some more development to deal with this in complex 
    - But in this limited time, just go for simple -->

### **Trimming the Series to their Sign start and end**

Using our detected start and end points, we trimmed each series of landmark data to only include the sign performance.

You can see the difference in distribution of the durations for the original and trimmed series, for each data source, in the boxplots below:

<div align="center">
<img src="assets/video_durations_orig_boxplot.png" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em>Boxplot of Duration by Data Source - Original Data</em></p>
</div>

<div align="center">
<img src="assets/video_durations_proc_boxplot.png" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em>Boxplot of Duration by Data Source - Preprocessed Data</em></p>
</div>

The difference in duration between the original and trimmed series was quite significant for some data sources. 
- The durations of both INES & UFV decreased significantly, and they ended up with much more similar distributions.
- The interquartile range of durations decreased quite a lot for INES, and decreased slightly for UFV and V-Librasil.
- SignBank already had very short durations, with little pause before and after the sign, so the durations only decreased slightly.
- V-Librasil had a wide range of long durations, and didn't decrease much.
  - This is probably because the videos appear to be in slow motion, with the signer moving very slowly. 
  - In real time, the sign duration, and the pause before and after, is more similar to the other data sources.

### **Scaling & Alignment**

The next 2 steps were unifying the scale and position alignment of the signers from across the data sources. They are separate steps but quite similar and related.

The basic process is to define some **reference points that represent the target scale / alignment**. Then by comparing the raw landmark data to the reference points, we can **calculate the scale factor and the horizontal & vertical shifts to be applied**.

**For example**, for the horizontal alignment:
- We want the signers to be horizontally in the center of the frame
  - So the target horizontal position is 0.5
- We use some heuristic to estimate the horizontal position of the signer:
  - We take the midpoint of the x values for the 2 shoulder landmarks
  - We take the midpoint of the x values for the leftmost and rightmost face landmarks
  - We take the average of the two midpoints to get a value representing the horizontal position of the signer
  - We do this for the whole series of frames and take the mean to get a value representing the horizontal position of the signer
- We then calculate the horizontal offset as the difference between 0.5 and the horizontal position of the signer
- We then apply the offset to the full series

An important point to mention, is that we calculate one set of transformation parameters per video series from aggregated landmark positions, then apply these same parameters to every frame. This preserves the relative motion within each video - signers aren't artificially recentered during signing.


<!-- The next 2 steps are unifying the scale and position alignment of the data. They are separate steps but quite similar and related.
The basic process is to define some reference points that represent the target scale / alignment.
plot?
then by comparing a samples points to the reference points, we can calculate the scale factor to be applied, and the horizontal and vertical shifts to be applied.
After some experiments and analysis, we decided to use these reference points/measurements as comparisons.
Ref pts
- we wanted to have the scale of the signer be as big as possible, so we lose as little info as possible by avoiding scaling down
- We also wanted the scale to not be so big we cut off some info
- from our analysis this scale results in the hands hardly ever being cut off from the edge
Comparison
- horizontal pos should be 0.5
    - take weighted avg of face midpoint and shoulder midpoint
- vertical pod should be approx A for shoulders midpoint and B for face midpoint
    - take weighted avg of the two companions to get the offset
- xscale
- yScale
An important point to mention is that for each series, we calculate these sfs and offsets based on an aggregate of the points on multiple frames, and then one sf/Forster value is applied to the full series. So within a series, the sense of scale & position is preserved. If the signer moves to the left in part of the video, we don't shift them back to the center for those frames.

Originally we used the median of all frames to calculate the reference points. The thinking was that the median would be less sensitive to outlier points due to something like above, the signer moving to the left. Or more commonly, their head tilting
- [insert highest variance videos vs lowest variance videos]
In the final iteration of the preprocessing, we moved to using the mean, not of the full series, but of just the frames before start/end frame.
Our logic was that these frames are supposed to be when the signer is most stationary. So they will provide a better reference point for approximating the signers base position. As expected the variance of these points is less then the full video. -->

### **Interpolating `None` frames**

#### **Context: `None` Landmarks in MediaPipe Output**
- Format of MediaPipe output
    - For a frame, individual landmarks can't be none. 
      - Only the full group of landmarks (face, pose, left hand, right hand) can be none
    - There are a few reasons to be `None`, and they also depend on the landmark type
- 99% of the time we have `None`s, they are hand landmarks 
    - A significant proportion of these are justified. The hand is not in the frame at the beginning or end of most videos in our dataset.
      - When the hand is outside the frame, the hand landmarks are `None`
    - Ignoring sequences of `None`s at the start / end, we still have quite a lot of `None`s
      - A significant proportion of these problematic `None`s are from the lowest resolution dataset, INES.
      - Even when they are in the frame, MediaPipe's confidence score is sometimes low enough, that the landmark is returned as `None`
        - We assume this is because the hand landmarks are quite detailed, so MediaPipe requires a higher resolution to detect them consistently.

#### **Remedy: Interpolation Process**

We developed a custom interpolation process to fill in the `None`s in the landmark data.
- For `None` landmarks at the start / end of the series, we applied a forward fill (repeating the first non-`None` landmark) and a backward fill (repeating the last non-`None` landmark)
- For `None` landmarks in the middle of the series, we applied a linear interpolation between the nearest non-`None` landmarks
- We also record the information about which frames were interpolated, and the degree of interpolation, so that we can pass it as a feature to the model



# **Model Development**
## **Overall Method: Landmark Feature Extraction -> Sequence Model**

For Sign Language recognition from video, the most conventional approach as of late is to extract features from each frame, treat the data as a time series, and use a model architecture that handles sequence data, like LSTM.

As discussed earlier in the report, we used pose estimation landmarks to develop the features for the model. For the model architecture, we experimented with RNN, LSTM, and Transformer and compared the results. 


## **Train / Test Set split & Cross Validation**

As much as we made an effort in preprocessing to remove significant differences between the data source like scale and position, some differences will still remain. Each signer will have their own style that has slightly different characteristics to the others- like speed movement, fluidity of movement, etc. 

So to make sure our model generalised, we stratified each data source, to make sure an equal proportion of each was in each training / validation / testing split. We also wanted to do 5-fold cross validation, again to make sure our model generalised well. 



<div align="center">
<p><h3><b><i>Train / Test Split</i></b></h3></p>
<img src="assets/train_test_split.svg" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p>With just 6 videos for each class, this meant dividing the training and testing sets like this.</p>
</div>



<div align="center">
<p><h3><b><i>Cross Validation Split</i></b></h3></p>
<img src="assets/train_val_split.svg" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p>And within the training set, dividing for 5-fold cross validation like this.</p>
</div>

We achieved this using scikit-learn's `StratifiedGroupKFold` class.
>This cross-validation object is a variation of StratifiedKFold [which] attempts to return stratified folds with non-overlapping groups. The folds are made by preserving the percentage of samples for each class in y in a binary or multiclass classification setting.\
>\- [StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

## **Frame Sampling**

We decided to implement **random frame sampling from each series** as part of the training process. This was to:
- Get a consistent sequence length for the input
- Reduce the computational cost of the model without losing too much information
- Act as a form of data augmentation to make the most of our small dataset

### **Why not use the full sequences?**

RNN, LSTM, and Transformer models can all **handle variable-length input** sequences if you use Padding + Masking, although each in slightly different ways. However the **variation in sequence lengths** in our dataset was very large.

The shortest series in our dataset was for `Cortar` from the `INES` data source, which was 21 frames long. The longest series was for `Sorvete` from the `V-Librasil` data source, which was 408 frames long.

There should be some natural variation in sequence length, each signer signs at different speeds. This will be somewhat associated with the data source, since most data sources had the same signer(s) for all their videos. But in our case, each data source also had **different framerates**, which has a significant impact on the sequence length. 


<!-- <div align="center">
<img src="assets/framecount_scatterplot.png" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
</div> -->

<!-- <div align="center">
<img src="assets/framecount_scatterplot_log.png" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
</div> -->

<div align="center">
<img src="assets/framecount_by_word.png" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p><em></em></p>
</div>

<!-- <div align="center">
<img src="assets/framecount_by_word_log.png" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
</div> -->


**Using the full series of frames for each data point would:**
- **Be more computationally expensive**
  - Particulary for Transformer, because every token 'attends' to every other token, meaning computation scales with O(n^2) where n is the sequence length
- **Have diminishing returns on longer sequences**
  - The difference between each frame at a higher framerate is much smaller, leading to repetitive information. It's likely we could skip some frames, and still have enough information to classify the sign.
  - With RNN & LSTM, past a certain length, earlier information is essentially 'forgetten' 
    - The gating mechanism in LSTM helps retain information for longer, but it it still finite
  - With Transformer, self-attention means the model can access all tokens directly, so there's no 'forgetten' information
    - But still, tasks have intrinsic context limits- past a certain length, extra tokens add noise instead of signal
- **Introduce data source specific bias**
  - With our small dataset, the model could learn to associate long sequences with a specific data source
  - For example, the `INES` data source always has the shortest sequences.
    - For the classes with their 1 `INES` series in the test set, the model could learn to disassociate long sequences with those classes during training.


### **How to sample the frames?**

**Set sample sequence length to 20 frames**
- We couldn't go much shorter without losing too much information.
  - The shortest series in our dataset was 21 frames (`Cortar` from the `INES` data source)
- We couldn't go much longer without having to repeat frames for series that were shorter than 20 frames.

**Randomly sample from a uniform distribution**
- Sometimes this technique is used with a normal distribution, focusing on the center of the sequence
- In our case, we had already trimmed the series to the sign performance, so we thought the uniform distribution would be more appropriate

**Sample multiple times from each series**
- In order to leverage the amount of information in the longer series, we decided to sample multiple times from each series.
- Each series was sampled up to 5 times, with a replacement rate of 0.2
- For shorter series, when there were insufficient remaining frames for a complete sample, they were combined with random frames that had already been sampled, to create one final complete sample
- This resulted in roughly 4.5 samples per series per epoch

**Resample at the start of each training epoch**
- The random sampling was performed at the start of each training epoch, using the epoch number as a random seed.
- This acts as a form of data augmentation
  - 1 series turns into ~4.5 series
  - Each epoch sees different frame combinations from the same videos
- Using the epoch number as the seed ensures reproducibility while still providing variety


## **Drawbacks**

**One drawback of this approach is that we lose temporal information.**
- We have already trimmed each series to the sign performance, and are sampling 20 frames from that
  - In theory, this means the speed of the sign is not preserved
  - All signs appear to take 20 frames to complete
- To remedy this, we included the original frame count and real-time duration of each series as features
- In future, we would like to combine the frame sampling with variable sequence length. For example:
  - Set a target framerate, and determine the number of frames to sample based on the source duration
  - So for 10fps, 30 frames are sampled from a 3 second sign, and 50 frames are sampled from a 5 second sign
  - This way we remove the data source bias, without losing the temporal information



## **Feature Engineering**
Using the **pose landmark features alone is not sufficient** for a model to understand the data. Since we understand what each pose landmark represents, we can imagine and engineer informative features from them. This is common practice when using pose landmarks to model Sign Languages. 

### **Types of Features**

We engineered three main categories of features from the MediaPipe pose and hand landmarks:

1. **Static Frame Features (Distances & Angles)**
   - Hand Features:
     - Inter-finger distances (e.g., fingertip-to-fingertip distances, finger base spread distances)
     - Finger joint angles (base, bend, and tip angles for each finger)
     - Inter-finger spread angles (e.g., thumb-index spread, index-ring spread)
   - Pose Features:
     - Hand-to-body distances (hands to head, shoulders, and cross-body measurements)
     - Arm joint angles (shoulder, elbow, wrist angles)
     - Upper body posture angles (shoulder tilt, neck tilt)

2. **Dynamic Frame-to-Frame Features (Landmark Motion Vectors)**
   - Hand motion vectors:
     - Wrist movement
     - Fingertip trajectories
     - Finger base point movements
   - Upper body motion vectors:
     - Shoulder and elbow trajectories
     - Wrist and hand landmark movements
     - Head/nose position changes

3. **Metadata Features (Some had different values frame-to-frame, some were constant across the frame series)**
   - Real-time duration between the detected motion start and end (constant)
   - The relative position of the frame in the full series (variable)
   - Mask indicating if the hand landmarks for the current frame are interpolated (variable)
   - Mask indicating the degree of interpolation for the current frame (variable)

### **Result**

<div align="center">
<img src="assets/feature_all_h.png" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p><em>Visualization of the features engineered for frame 18 of the sign 'cortar' (SignBank data source)</em></p>
</div>

All features were computed in 2D space and normalized appropriately to ensure consistency across different video sources and signers. The combination of these feature types allows the model to capture both the static pose information and the dynamic aspects of sign language gestures.

**The resulting number of features for each type was:**
- 50 landmark position coordinates
- 33 distances between landmarks in a frame
- 86 angles between landmarks in a frame
- 62 movements between landmarks in consecutive frames
- 8 features representing various metadata


## **Models**
## **Training Process**
As this is an unfounded, open source project, we didn't have convenient access to GPUs for training. 
<!-- The training code was developed to be platform agnostic, but training on GPU was ~X times faster, taking the typical training process from ~Y hours to ~Z hours depending on the model architecture being used.  -->
And as we are collaborating internationally, we needed to be able to track experiments results in one place. 
We considered using a tool like DVC, but it typically requires setting up paid remote storage. 

The solution we decided on was using Google Colab.  It would be cost effective as even free accounts can access GPU runtimes. And it would be time effective as it is relatively easy to set up. 

The setup consisted of developing a notebook with cells to:
- Install any necessary dependencies
- Clone the repository code into the runtime
- Mount the project google drive folder to the session storage
- Allow the user to easily edit key config params
- Begin the training process with live monitoring
- Log epoch results and best model files directly to a runs folder on google drive
- Be able to resume interrupted runs from the same place with the same environment, and switch between GPU or CPU
  - Important as Colab reserves the right to disconnect your runtime for a variety of reasons, even with Pro, and GPU usage is limited


# **Results**

## **Experiments**

We executed multiple training experiments, with different model types, input features, and data augmentation techniques.

### **Model Types**

We ran experiments with these three model types and configurations:

**RNN**
- 2 layers
- 256 hidden units

**LSTM**
- 2 layers
- 256 hidden units

**Transformer**
- 2 encoder layers
- 128-dimensional hidden size
- 8 attention heads
- Feedforward hidden size: 256

### **Input Features**

We ran experiments with 2 different sets of input features:
- Including the 150 landmark position coordinates
  - 339 input features for each frame
- Excluding the 150 landmark position coordinates
  - 189 features for 20 frames per sample

### **Data Augmentation**

All experiments used the same data augmentation settings:

- Rotation (+- 10 degrees)
  - 0.5 probability of applying the rotation
- Noise (0.05 std)
  - 0.5 probability of applying the noise

### **Training Configuration**

All experiments used the same training configuration:

- 300 epochs
- 64 batch size
- 5-fold cross validation
- AdamW optimizer
- ReduceLROnPlateau learning rate scheduler
- Early stopping with patience = 20

## **Summary of results**

<table style="font-size: medium; text-align: center; table-layout: fixed; width: 100%;">
  <colgroup>
    <col style="width: 16%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
  </colgroup>
  <thead>
    <tr>
      <th>Model Type</th>
      <th>No. of Features</th>
      <th>Loss</th>
      <th>Accuracy </th>
      <th>Top-2 Accuracy</th>
      <th>Top-3 Accuracy</th>
      <th>Top-4 Accuracy</th>
      <th>Top-5 Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RNN</td>
      <td>189</td>
      <td>2.802</td>
      <td>51.33%</td>
      <td>70.80%</td>
      <td>75.22%</td>
      <td>85.84%</td>
      <td>89.38%</td>
    </tr>
    <tr>
      <td>RNN</td>
      <td>339</td>
      <td>2.673</td>
      <td>62.83%</td>
      <td>81.42%</td>
      <td>86.73%</td>
      <td>92.04%</td>
      <td>94.69%</td>
    </tr>
    <tr>
      <td><b>LSTM</b></td>
      <td><b>189</b></td>
      <td><b>2.664</b></td>
      <td><b>66.37%</b></td>
      <td><b>77.88%</b></td>
      <td><b>84.07%</b></td>
      <td><b>89.38%</b></td>
      <td><b>93.81%</b></td>
    </tr>
    <tr>
      <td>LSTM</td>
      <td>339</td>
      <td>2.694</td>
      <td>63.72%</td>
      <td>75.22%</td>
      <td>84.96%</td>
      <td>93.81%</td>
      <td>95.58%</td>
    </tr>
    <tr>
      <td>Transformer</td>
      <td>189</td>
      <td>2.715</td>
      <td>61.06%</td>
      <td>75.22%</td>
      <td>84.96%</td>
      <td>86.73%</td>
      <td>92.92%</td>
    </tr>
    <tr>
      <td>Transformer</td>
      <td>339</td>
      <td>2.695</td>
      <td>60.18%</td>
      <td>81.42%</td>
      <td>86.73%</td>
      <td>90.27%</td>
      <td>91.15%</td>
    </tr>
  </tbody>
</table>
<br/>

**LSTM models had the best results**, although the difference in performance between the 3 model types is not so significant.

Most models have **Top-5 Accuracy > 90%**, but we care more about the basic accuracy.

For LSTM & Transformer, removing position features actually **improves test performance**. This is probably because removing them causes the model to overfit less, since for all models, including the position features resulted in lower Training Loss.




## **Best Model Results**

The best performing model was the **LSTM model with 189 features**

### **Overfitting**

The training loss and validation loss for this model were `0.02336` and `0.00660` respectively. These are significantly lower than the test loss, **indicating overfitting**. 

We took **measures to prevent overfitting**: like randomly sampling frames, applying data augmentation, and stratifying the data sources in each split. The data augmentation is the reason the training loss is higher than the validation loss. However the model still overfits in the end. 

We expect that the small dataset size is a large reason the model overfits. To remedy this, a larger dataset would of course help, but in lieu of that, we could apply **more aggresive data augmentation** to help the model generalize better to new features in unseen data.

### **Misclassification**
 
Inspecting the results on the test set in more detail, we can which signs are being consistently misclassified by the model. Random sampling of 20-frame sequences was done on the test set too. So although there was just one source video for each sign, these results are based on the model's predictions on multiple samples from each.

<div align="center">
<img src="assets/confusion_matrix.png" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p><em></em></p>
</div>

**Examples of misclassifications:**
- `banana` and `ajudar` were usually misclassified as `ano`
- `casa` and sometimes `cafe` were misclassified as `familia`
- `sopa` was always misclassified, as either `animal` or `cafe`

<!-- <div align="center">
<img src="assets/probability_heatmap.png" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p><em>Figure 1: A description of the image</em></p>
</div> -->

<!-- ### **Feature Importance** -->






  
## **Future ideas**

We are very proud of what we managed to achieve in such a short time with this project. We were able to develop a model with performance matching similar projects in the LIBRAS domain. But we also saw many opportunities for improvement, giving us lots of promising ideas for future improvements.

### **Run a wider range of experiments**

We would leverage the power of our Hydra configured training pipeline to find the best combination of model architecture and hyperparameters:
- By gridsearching ranges of settings and parameters we think will address the issues
- By using Hydra's integration with Optuna to intelligently search for the best settings and parameters

### **Further develop the feature engineering process**

We can already see that the features we engineered are informative, because the performance was often better when we relied on them over the raw pose landmarks.
- We would like to engineer more features, and test their performance
- We would also like to explore the feature importance to better understand which are most informative
- We only used the 2D landmark coordinates, but our codebase is set up to easily use the 3D landmark coordinates too

### **Further develop the data augmentation process**
Since we experienced quite a difference in performance between the training and testing environments, we would at least like to experiment with more variety in the data augmentation process
- At least try increasing the probability of application for more aggressive data augmentation
- Try different ranges for the rotation and noise
- Apply some new types of data augmentation

### **Expand the dataset**

We had 25 classes in our dataset. The signs for some of these were quite similar, resulting in some misclassification.

We would like to expand the dataset to include more classes. Some distinct signs could improve the scope of our model without hurting overall performance. It could also potentially help the model generalize better by seeing more variety in the data.

### **More sophisticated frame sampling**

We would further develop the frame sampling implementation to include variable sequence length:
- Set a target framerate, and determine the number of frames to sample based on the source duration
- This would help to retain temporal info like speed of sign / amount of movement between each frame

# **Demo Application Development**

# **Contributors**

The main work for this project took place over 4 months, from February to June 2025. Below is a list of the people that contributed to the project. 

Feel free to reach out to them if you have questions about any aspect of the project. Some members have also made additional changes & improvements since the end of the main project period.

### **Project Leader**
<p><a href="https://www.linkedin.com/in/ben-d-thompson/"><strong>Ben Thompson</strong></a></p>
<ul>
<li>Tasks: Research Resources, Data Scraping, Data Preprocessing, Model Development, Demo App Development</li>
<li>Omdena Role: Project Leader & Lead ML Engineer</li>
</ul>

### **Task Leaders**

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">

<div>

<p><a href="https://www.linkedin.com/in/ayushya-pare/"><strong>Ayushya Pare</strong></a></p>
<ul>
<li>Tasks: Research Resources, Data Scraping</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/gustavopsantos"><strong>Gustavo Santos</strong></a></p>
<ul>
<li>Tasks: Data Scraping, Data Cleaning & Organisation</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

</div>

<div>

<p><a href="https://www.linkedin.com/in/anastasiia-derzhanskaia/"><strong>Anastasiia Derzhanskaia</strong></a></p>
<ul>
<li>Tasks: Model Development</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/patrick-fitz-b2186a11b/"><strong>Patrick Fitz</strong></a></p>
<ul>
<li>Tasks: Demo App Development</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

</div>

</div>

### **Task Contributors**

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">

<div>

<p><a href="https://www.linkedin.com/in/michael-spurgeon-jr-ab3661321/"><strong>Michael Spurgeon Jr</strong></a></p>
<ul>
<li>Tasks: Data Scraping, Model Development</li>
<li>Omdena Role: ML Engineer</li>
</ul>

<p><a href="https://github.com/prajinip13"><strong>Pooja Prasad</strong></a></p>
<ul>
<li>Tasks: Data Review & Cleaning</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/ethel-phiri261312/"><strong>Ethel Phiri</strong></a></p>
<ul>
<li>Tasks: Data Scraping</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/damla-helvaci/"><strong>Damla Helvaci</strong></a></p>
<ul>
<li>Tasks: Data Review & Cleaning</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

</div>

<div>

<p><a href="https://www.linkedin.com/in/wafa-basoodan-9447616a/"><strong>Wafa Basudan</strong></a></p>
<ul>
<li>Tasks: Model Development</li>
<li>Omdena Role: ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/kunal-sood-3b993225b/"><strong>Kunal Sood</strong></a></p>
<ul>
<li>Tasks: Model Development</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

<p><a href="https://github.com/guluzar-gb"><strong>Gulzar Helvaci</strong></a></p>
<ul>
<li>Tasks: Data Review & Cleaning</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

</div>

</div>