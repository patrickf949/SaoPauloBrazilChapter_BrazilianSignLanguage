<mark>This page is under construction</mark>

# **Sign Language Translation for Brazilian Healthcare - Omdena Project Page**

## **Background**
This project was done as part of an Omdena local chapter challenge with the Sao Paulo chapter. These challenges are open source collaborations between volunteer developers around the world, working to create AI application that solve social issues for local communities. 

Brazil has a significant deaf community, with Brazilian Sign Language (Libras) recognized as an official language. However, communication barriers often exist in healthcare settings, potentially compromising the quality of care for deaf patients. This potentially lead to misdiagnoses, treatment errors, or a compromised patient experience.

This project aimed to address this by developing a sign language translation model for Brazilian Sign Language (LIBRAS), and a demo web app to showcase the model.

## **Project Summary**

### Domain - Sign Language Processing
https://research.sign.mt/

### Outline:
- Scraped data from 4 different sources
- Cleaned the data to create a dataset of 1000+ videos of LIBRAS signs
- Reviewed the data to find words with the most videos
- Created a target dataset of 25 words with high quality videos, and words related to the healthcare domain
- Developed an advanced preprocessing pipeline that allowed us to use data from multiple sources
- Developed a modelling code base that allowed us to experiment with different data sampling, data augmentation, model architectures and training strategies
- Ran a range of model experiments, and found the best performing model
- Developed a demo web app to showcase the model
- Created a report summarizing the project and the results

### Results
- a
- b
- c

### Deliverables
The deliverables for this project consisted of a live demo application, the open source code, and a report of our work and results.

You can see the model in action and test it on your own videos in the [demo app](link-to-be-added).

You can find the code for the project and instructions for setup/usage in our Github Repository: [https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage](https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage)

You can read more details about the project in the report below: [Jump to the report](#report)


## Contributors

The main work for this project took place over 4 months, from February to June 2025. Below is a list of the people that contributed to the project. Feel free to reach out to them if you have questions about any aspect of the project. Some members have also made additional changes & improvements since the end of the main project period.

### **Project Leader**

[Ben Thompson](https://www.linkedin.com/in/ben-d-thompson/) 
- Tasks: Research Resources, Data Scraping, Data Preprocessing, Model Development, Demo App Development
- Omdena Role: Project Leader & Lead ML Engineer

### **Task Leaders**
<table>
<tr>
</tr>
<tr>
<td>

[**Ayushya Pare**](https://www.linkedin.com/in/ayushya-pare/)
- Tasks: Research Resources, Data Scraping
- Omdena Role: Lead ML Engineer

[**Gustavo Santos**](https://www.linkedin.com/in/gustavopsantos) 
- Tasks: Data Scraping, Data Cleaning & Organisation
- Omdena Role: Lead ML Engineer
</td>
<td>

[**Anastasiia Derzhanskaia**](https://www.linkedin.com/in/anastasiia-derzhanskaia/) 
- Tasks: Model Development
- Omdena Role: Lead ML Engineer

[**Patrick Fitz**](https://www.linkedin.com/in/patrick-fitz-b2186a11b/) 
- Tasks: Demo App Development
- Omdena Role: Lead ML Engineer
</td>
</tr>
</table>

### **Task Contributors**
<table>
<tr>
</tr>
<tr>
<td>

[**Michael Spurgeon Jr**](https://www.linkedin.com/in/michael-spurgeon-jr-ab3661321/) 
- Tasks: Data Scraping, Model Development
- Omdena Role: ML Engineer

[**Pooja Prasad**](https://github.com/prajinip13) 
- Tasks: Data Review & Cleaning
- Omdena Role: Junior ML Engineer
</td>
<td>

[**Wafa Basudan**](https://www.linkedin.com/in/wafa-basoodan-9447616a/) 
- Tasks: Model Development
- Omdena Role: ML Engineer

[**Kunal Sood**](https://www.linkedin.com/in/kunal-sood-3b993225b/) 
- Tasks: Model Development
- Omdena Role: Junior ML Engineer
</td>
<td>

[**Ethel Phiri**](https://www.linkedin.com/in/ethel-phiri261312/) 
- Tasks: Data Scraping
- Omdena Role: Junior ML Engineer

[**Gulzar Helvaci**](https://github.com/guluzar-gb) 
- Tasks: Data Review & Cleaning
- Omdena Role: Junior ML Engineer
</td>
</tr>
</table>

[**Damla Helvaci**](https://www.linkedin.com/in/damla-helvaci/) 
- Tasks: Data Review & Cleaning
- Omdena Role: Junior ML Engineer
------------------



# **Report**

## **Contents**
- 

## **Introduction**
### Problem statement
*Similar to what we could find in the [Omdenas website](https://www.omdena.com/chapter-challenges/ai-assisted-sign-language-translation-for-brazilian-healthcare-settings)*

The problem systems we started the project with is here.

The scope of this is quite broad and ambitious for a 3-month project. While still aiming high, we kept this in mind to inform our decision making. For example, when deciding which words in the dataset to focus on, we included words that would be most likely to be used in a medical context. And we worked with the thinking that this is an initial proof-of-concept, of a solution we would like to develop further in future.


##  **Research & Planning**

### Research
To decide on the scope of the project and our plan for development, we began with research. 

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
    
#### Domain: Sign Language Processing (SLP)

Sign Language Processing (SLP) is a field of artificial intelligence that combines Natural Language Processing (NLP) and Computer Vision (CV) to automatically process and analyze sign language content. Unlike spoken languages that use audio signals, signed languages use visual-gestural modality through manual articulations combined with non-manual elements like facial expressions and body movements.

**Key SLP Tasks Relevant to Our Project:**

1. **Sign Language Recognition**: Identifying individual signs or sequences of signs from video input
2. **Sign Language Translation**: Converting sign language videos into spoken language text (our primary focus)
3. **Sign Language Production**: Generating sign language videos from spoken language text
4. **Sign Language Detection**: Determining if a video contains sign language content
5. **Sign Language Segmentation**: Identifying boundaries between different signs in continuous signing

**Challenges in SLP:**
- **Visual-gestural modality**: Unlike spoken languages, signed languages lack a written form, forcing researchers to work directly with raw video signals
- **Simultaneity**: Multiple articulators (hands, face, body) can convey information simultaneously
- **Spatial coherence**: Spatial relationships and movements are crucial for meaning
- **Lack of standardization**: Limited annotated datasets, especially for languages like Brazilian Sign Language (Libras)

*Reference: [Sign Language Processing Star](https://research.sign.mt/) - A comprehensive resource for SLP research and datasets*


#### Data Sources: LIBRAS Datasets

##### Review of available datasets
We surveyed existing LIBRAS datasets, and reviewed their contents. This table contains details from the 9 most relevant datasets we found.

| Dataset | Year | Type | Format | Access | Number of Classes | Examples per Class | Total Examples |
|---------|------|---------------|---------|---------|----------------|-------------|-------------|
| Brazilian Sign Language Words Recognition Dataset | - | Words | Images | [Downloadable](https://www.kaggle.com/datasets/alvarole/brazilian-sign-language-words-recognition) | 15 | ~60 - 1000 | ~5000 |
| Brazilian Sign Language Alphabet Dataset | 2020 | Alphabet | Images | [Downloadable](https://biankatpas.github.io/Brazilian-Sign-Language-Alphabet-Dataset/?utm_source=chatgpt.com) | 15 | 150 - 600 | 3000 |
| Libras Movement | 2009 | 'Hand Movement Types' | Hand Landmarks | [Downloadable](https://archive.ics.uci.edu/dataset/181/libras+movement) | 15 | - | - |
| Libras SignWriting Handshape (LSWH100) | 2024 | 'Handshapes' | CG Images | [Downloadable](https://www.sciencedirect.com/science/article/pii/S2352340924007455) | 100 | ~480 | 48000 |
| LIBRAS Cross-Dataset Study | 2023 | Words | Videos | [Restricted](https://github.com/avellar-amanda/LIBRAS-Translation?tab=readme-ov-file) | 49 | ~6 | ~294 |
| **V-Librasil - A New Dataset with Signs in Brazilian Sign Language (Libras)** | - | Words | Videos | [Scrapable](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras) | 1,364 | 3 | 4089 |
| **Federal University of Viçosa (UFV) – LIBRAS-Portuguese Dictionary Dataset** | - | Words | Videos | [Scrapable](https://sistemas.cead.ufv.br/capes/dicionario/) | 1,004 | 1 - 2 | 1029 |
| **National Institute of Deaf Education (INES) – LIBRAS Dictionary Dataset** | - | Words | Videos | [Scrapable](https://www.ines.gov.br/dicionario-de-libras/) | 237 | 1 - 2 | 282 |
| **SignBank - LIBRAS Dataset (Universidade Federal de Santa Catarina)** | - | Words | Videos | [Scrapable](https://signbank.libras.ufsc.br/en) | 3090 | 1 | 3090 |

##### Datasets selected for our task

We decided that although it is more difficult, we wanted to focus on sign **videos**, not images. 
 - This would be more applicable to a real-world situation.

We also decided to focus on sign **words**, not alphabet signs, movement/shape types, or sentences. 
 -  Again, **words** would be more applicable to a real-world situation than alphabet signs or movement/shape types. 

Sign sentences would be the most similar to real-world use, and also the most challenging to model accurately. 
  - However we didn't find any datasets with this type of data. 
  - Even in SLP more broadly, large high quality datasets of sentences are rare, so it is not surprising we couldn't find any for LIBRAS.



#### Exisiting Literature: Sign Language Processing with LIBRAS Data

We found a few papers working with LIBRAS data, but most were not directly relevant to our project, as they focussed on different tasks, like images instead of videos, or classification of movement types instead of words.

The most relevant paper we found was the 2023 paper [A Cross-Dataset Study on the Brazilian Sign Language Translation](https://openaccess.thecvf.com/content/ICCV2023W/CLVL/papers/de_Avellar_Sarmento_A_Cross-Dataset_Study_on_the_Brazilian_Sign_Language_Translation_ICCVW_2023_paper.pdf) by de Avellar & Sarmento.

This paper was a cross-dataset study on the Brazilian Sign Language Translation task, and it was very relevant to the goals of our project. 

They combined the 4 datasets we selected for our task, and used an LSTM model to classify the signs. They tried two different methods for feature extraction: pre-trained Convolutional Neural Network (CNN) models, and Landmark Estimation. 

This was a good reference point for us to confirm our plan and approach was sound. We investigated if any new datasets had become available since the publishing, or if any existing had been expanded. Based on our review of wider SLP lit, we concluded that the modelling approach they followed was still appropriate for us. We aimed to differentiate our approach from theirs with more detailed preprocessing to address the data source variation, expanding our data augmentation and engineering new features. 

They made a lot of effort to scrape the data, clean, and preprocess it into one unified dataset. This is the 2023 'LIBRAS Cross-Dataset Study' dataset listed in the table above. Kindly the dataset is available on request, but unfortunately we didn't hear back from one of the authors. So we would have to scrape, clean, and preprocess the data ourselves.

### Plan
#### Overview
- We will develop AI models to classify videos of LIBRAS word signs
    - Not images of signs, and not alphabet signs
- If possible, we will focus on healthcare related words, in line with the original goal
    - But since we have to be realistic with the data available to us, and this is just a PoC, the scope is flexible
    - If the data is limited in quantity or quality, we will focus on general words
- 
- We will follow a similar methodology to:
    - 2023 paper collecting public LIBRAS datasets & applying SLT
    - Key details and points:
        - A Cross-Dataset Study on the Brazilian Sign Language Translation (2023)
    - Omdena Indonesia’s Sign Language Translation Project
    - Key details and points:
        - Omdena Indonesia - Report Summary
- To understand our goal in this project, it would be helpful to read about them.
    - 2023 paper - A Cross-Dataset Study
#### Data

We will scrape videos of LIBRAS words from the following 4 data sources:
* [INES](https://www.ines.gov.br/dicionario-de-libras/)
* [Signbank (UFSC)](https://signbank.libras.ufsc.br/pt)
* [UFV Dictionary](https://sistemas.cead.ufv.br/capes/dicionario/)
* [V-Librasil Dataset](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras#files)


#### Method

#### Tasks



#### Deliverables


We will try to use the same 4 source datasets as them
We will try to also explore both CNN & Landmark Based feature extraction methods
Omdena Indonesia’s Sign Language Translation Project
We might record our own videos like they did
We will try to also explore both CNN & Landmark Based feature extraction methods
We will try to create a similar Report & Demo App 

## Data Collection

To build a robust dataset of BSL videos, we will develop code to scrape videos and metadata from the following 4 data sources:
* [INES](https://www.ines.gov.br/dicionario-de-libras/)
* [Signbank (UFSC)](https://signbank.libras.ufsc.br/pt)
* [UFV Dictionary](https://sistemas.cead.ufv.br/capes/dicionario/)
* [V-Librasil Dataset](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras#files)

We will then clean the metadata from each data source, and process them into a unified format for easy management and analysis of the dataset.

We will define some criteria to narrow down the dataset, manually review the subset of potentially usable videos, and finalize our project dataset.

### Scraping

Due to the scale and structure of these 4 data sources, we implemented web automation tools—primarily **Selenium**—to efficiently extract video URLs and relevant metadata. For efficient processing, we initally just scraped metadata and download URLs for each video, rather than directly downloading thousands of video files that we might not use. 

The scraped data from each source was compiled into a CSV file, which served as a foundational resource for later stages of data cleaning, preprocessing, and modeling.

We scraped data from three Brazilian Sign Language (Libras) resources: INES, Signbank (UFSC), and VLibras. We used Python with Selenium to automate browsing and extract videos, words, and metadata. Each website had a different structure, so we customized the code for each. On INES, videos were downloaded directly. UFSC’s Signbank had scattered links, which we cleaned using pandas. VLibras was slower and more complex, so we scraped it manually with help from browser tools. We saved all data in organized CSV files.

**Web Automation Approach:**
We used Selenium to automate the collection process, enabling consistent extraction of video files and associated metadata across varying site structures.

**Extracted Metadata Fields:**
The specific metadata varied slightly by source, but key fields included:

* `alphabet_label`: the starting letter of the signed word
* `label`: the word being signed in the video
* `video_url`: direct link to the video file

This standardized structure allowed us to create a unified dataset for downstream processing.

### Cleaning

With the metadata for each data source collected,  we began the cleaning step. This task should solve and help with a few issues:

1. **Homographs:** Like any other language, there are some cases of homographs in Portuguese, that is, words that have the same spelling and different meanings, an example of homograph in English is the word "bat" that could either refer to the animal or the object used to hit the ball in a baseball game. Due to the structure of the data, these words would be registered with the same label but have different signs.
2. **Sign synonyms:** Some words could have more than one sign, depending on the region or signer. Similarly to the homographs, these words would be registred under the same label but have different signs in the videos.
3. **Lack of video examples:** Even though there were videos for every registered word in the datasets, some of them had fewer videos than others. For the project to be successful, it was necessary to have at least a certain amount of example videos to train the models.
4. **Healthcare related words:** The project's main goal was to act on translating LIBRAS specifically in healthcare settings. Therefore, it was necessary to select health related words to compose a target dataset.

Below, you can find the approach used to solve each of the issues above.

#### Video reviewing
[comment]: <TODO: continue video reviewing and data cleaning steps from here>


### Review steps

### Final dataset

![alt text](assets/all_signs_for_banana.gif "placeholder")

abc

<img src="assets/all_signs_for_banana.gif" alt="isolated" width="200"/>

## Data Preprocessing
<!-- 
![alt text](assets/placeholder.png "placeholder")


<img src="assets/placeholder.png" alt="isolated" width="200"/>

<img src="assets/placeholder.png" alt="isolated" title="hover hint" style="width: 75%;"/> -->

### EDA

- Dataset differences
    - dimensions
        - grid of data sources & dims
    - frame rates
        - on same grid as above?
    - durations
        - plot for each label



#### Dimensions & FPS

<img src="assets/fps_bar_chart.png" alt="isolated" title="hover hint" style="width: 75%;"/>

<img src="assets/video_dimensions_for_animal.png" alt="isolated" title="hover hint" style="width: 75%;"/>

<img src="assets/video_dimensions_bar_chart.png" alt="isolated" title="hover hint" style="width: 75%;"/>


#### Durations

### Pose estimation with [MediaPipe Holistic](https://ai.google.dev/edge/mediapipe/solutions/guide)
why:
    - will be used in preprocessing for motion detection, offset & scaling
    - Will be used as the base features that will be input to the model

<img src="assets/pose_for_casa_on_orig_and_black_raw.gif" alt="isolated" title="hover hint" style="width: 75%;"/>



### Start/End Point Trimming

our first preprocessing step is to trim the videos. We will remove the periods from the start and end of the video where the signer is stationary. Resulting in shorter videos where to the actual sign performance takes up the majority of the time.

#### Motion Detection 

- to do this we explored various different methods for measuring motion between frames:
    - bg sub
    - Basic
    - Landmarks
- Each normalised and moving avgd for smooth results, and consistency between datasets
    - we are mainly interested in peaks, so normalising makes sense
    - We don’t have some sense of the correct ’absolute value’ across datasets
- Show some before & after
- show some example via gifs
- for some previous versions, we used a combination of multiple, but in the final version settled on just landmarks


<img src="assets/motions_for_aniversario.gif" alt="isolated" title="hover hint" style="width: 75%;"/>

#### Analysing Motion Start and End

- basic method is just taking an absolute threshold for the start & end
- complex method involves 
    - detecting the first and last peak to get the correct general location 
    - taking the rate of change of the motion 
    - Search back/forward from the peaks for the index where the slope has an inflection point 
        - this should show when the movement really started 
    - Go 0.X seconds before this as a buffer
        - seconds rather than frames to be robust to different datasets
- in the final version we just went for the simple version 
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
    - But in this limited time, just go for simple



<img src="assets/video_durations_orig_boxplot.png" alt="isolated" title="hover hint" style="width: 75%;"/>
<img src="assets/video_durations_proc_boxplot.png" alt="isolated" title="hover hint" style="width: 75%;"/>

### Scaling & Alignment

The next 2 steps are unifying the scale and position alignment of the data. They are separate steps but quite similar and related.
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
An important point to mention is that for each series, we calculate these sfs and offsets based on an aggregate of the points on multiple frames, and then one sf/Forster value is applied to the full series. So within a series, the sense of scale & position is preserved. If the signer moves to the left in part of the video, we don’t shift them back to the center for those frames.

Originally we used the median of all frames to calculate the reference points. The thinking was that the median would be less sensitive to outlier points due to something like above, the signer moving to the left. Or more commonly, their head tilting
- [insert highest variance videos vs lowest variance videos]
In the final iteration of the preprocessing, we moved to using the mean, not of the full series, but of just the frames before start/end frame.
Our logic was that these frames are supposed to be when the signer is most stationary. So they will provide a better reference point for approximating the signers base position. As expected the variance of these points is less then the full video.

### Interpolating `none` frames

Context
- Format of MediaPipe output
    - For a frame, individual landmarks can't be none. Only the full group of landmarks can be none
    - There are a few reasons to be None, and they also depend on the type
- 99% of the time we have Nones, they are hand landmarks being None
    - This is because of how they are detected
    - A significant proportion of these are justified, the hand is not in the frame at the beginning or end
        - Ignoring sequences of Nones at the start / end, we still have quite a lot of Nones
        - (Plot showing None sequences)
    - A significant proportion of these problematic Nones are from the lowest resolution dataset, INES 

### Summary

<img src="assets/pose_for_casa_on_black_preproc.gif" alt="isolated" title="hover hint" style="width: 75%;"/>

## Model Development
### Landmark -> LSTM method
For Sign Language recognition from video, the most common  approach as of late is to extract features from each frame, treat the data as a time series, and use a model like LSTM.

#### Overview
#### Train / Validation / Test split
- diagram from slides
#### Feature Engineering
Using the pose landmark features alone is not sufficient for a model to understand the data. Since we understand what each pose landmark represents, we can imagine and engineer informative features from them. This is common wisdom when using pose landmarks to model Sign Languages. 

We engineered the following features directly from a single 
frame of pose landmarks, to better represent the information 
in an individual frame. :
- angle of the line between 3 landmarks representing:
    - wrist & finger positions
    - arm position
- distance between 2 landmarks representing:
    - wrist & finger positions
    - arm position

We engineered three main categories of features from the MediaPipe pose and hand landmarks:

1. **Static Frame Features (Angles & Distances)**
   - Hand Features:
     - Finger joint angles (base, bend, and tip angles for each finger)
     - Inter-finger spread angles (e.g., thumb-index spread, index-ring spread)
     - Inter-finger distances (e.g., fingertip-to-fingertip distances, finger base spread distances)
   - Pose Features:
     - Arm joint angles (shoulder, elbow, wrist angles)
     - Hand-to-body distances (hands to head, shoulders, and cross-body measurements)
     - Upper body posture angles (shoulder tilt, neck tilt)

2. **Dynamic Features (Frame-to-Frame Differences)**
   - Hand motion vectors:
     - Wrist movement
     - Fingertip trajectories
     - Finger base point movements
   - Upper body motion vectors:
     - Shoulder and elbow trajectories
     - Wrist and hand landmark movements
     - Head/nose position changes

3. **Global Features (Sequence-Level Statistics)**
   - Movement range statistics
   - Velocity and acceleration profiles
   - Spatial distribution metrics of hand positions
   - Temporal pattern features

All features were computed in 2D space and normalized appropriately to ensure consistency across different video sources and signers. The combination of these feature types allows the model to capture both the static pose information and the dynamic aspects of sign language gestures.

We engineered the follow features across sequential frames, 
to better represent the information and relationship across 
frames:
- angle of the line between 3 landmarks representing:
    - wrist & finger positions
    - arm position
- distance between 2 landmarks representing:
    - wrist & finger positions
    - arm position
We added the following features from across the whole frame 
series that would be constant across frame’s feature vector 
in an individual series. The goal of these was to represent 
traits about the whole serie

#### Data Augmentation
- diagram from slides??
#### Models
#### Training Process
As this is an unfounded, open source project, we didn’t have convenient access to GPUs for training. The training code was developed to be platform agnostic, but training on GPU was ~X times faster, taking the typical training process from ~Y hours to ~Z hours depending on the model architecture being used. 
And as we are collaborating internationally, we needed to be able to track experiments results in one place. 
We considered using a tool like DVC, but it typically requires setting up paid remote storage. 

The solution we decided on was using Google Colab.  It would be cost effective as even free accounts can access GPU runtimes. And it would be time effective as it is relatively easy to set up. The setup consisted of developing a notebook with cells to:
install any necessary dependencies
clone the repository code into the runtime
mount the project google drive folder to the session storage
Allow the user to easily edit key config params
Begin the training process with live monitoring
Log epoch results and best model files directly to a runs folder on google drive
Be able to resume interrupted runs from the same place with the same environment, and switch between GPu or cpu (v important as Colab reserves the right to disconnect your runtime for a variety of reasons, even with Pro)


## Results
### Overview
### Analysis
### Future ideas
leverage the power of hydra to Gridsearch exp params
Leverage the power of hydra and optuna to find best prams 


## Demo Application Development