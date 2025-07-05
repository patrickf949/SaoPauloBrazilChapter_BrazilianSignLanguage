<mark>This page is under construction</mark>

# Sign Language Translation for Brazilian Healthcare

## Background / Introduction / no heading

This project was done as part of an Omdena local chapter challenge with the Sao Paulo chapter. These challenges are open source collaborations between volunteer developers around the world, working to create AI application that solve social issues for local communities. 

Brazil has a significant deaf community, with Brazilian Sign Language (Libras) recognized as an official language. However, communication barriers often exist in healthcare settings, potentially compromising the quality of care for deaf patients. This potentially lead to misdiagnoses, treatment errors, or a compromised patient experience.

This project aimed to address this by developing a sign language translation model for Brazilian Sign Language (LIBRAS), and a demo web app to showcase the model.

## Project work:
- Scraped data from 4 different sources
- Cleaned the data to create a dataset of 1000+ videos of LIBRAS signs
- Reviewed the data to find words with the most videos
- Created a target dataset of 25 words with high quality videos, and words related to the healthcare domain
- Developed an advanced preprocessing pipeline that allowed us to use data from multiple sources
- Developed a modelling code base that allowed us to experiment with different data sampling, data augmentation, model architectures and training strategies
- Ran a range of model experiments, and found the best performing model
- Developed a demo web app to showcase the model
- Created a report summarizing the project and the results

You can see the model in action in the [demo app](link-to-be-added).

You can find the code for the project in our Github Repository: [https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage](https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage)

You can read more details about the project in the report below: [Report Beginning](#report)


## Results

## Contributors

This project took place over 4 months, from February to June 2025. Here's a list of people that contributed to the project. Feel free to reach out to them if needed:

**Project Leader**

[Ben Thompson](https://www.linkedin.com/in/ben-d-thompson/) (Project Leader & Lead ML Engineer)
- Main tasks: Data Scraping, Data Preprocessing, Modelling, Demo Web App

**Task Leaders**

[Ayushya Pare](https://www.linkedin.com/in/ayushya-pare/) (Lead ML Engineer)
- Main tasks: Research Resources, Data Scraping

[Gustavo Santos](https://www.linkedin.com/in/gustavopsantos) (Lead ML Engineer)
- Main tasks: Data Scraping, Data Cleaning & Organisation

[Anastasiia Derzhanskaia](https://www.linkedin.com/in/anastasiia-derzhanskaia/) (Lead ML Engineer)
- Main tasks: Model development

[Patrick Fitz](https://www.linkedin.com/in/patrick-fitz-b2186a11b/) (Lead ML Engineer)
- Main tasks: App development

**Task Leaders**
<table>
<tr>
</tr>
<tr>
<td>

[**Ayushya Pare**](https://www.linkedin.com/in/ayushya-pare/) 
- Role: Lead ML Engineer
- Tasks: Research Resources, Data Scraping

[**Gustavo Santos**](https://www.linkedin.com/in/gustavopsantos) (Lead ML Engineer)
- Main tasks: Data Scraping, Data Cleaning & Organisation
</td>
<td>

[**Anastasiia Derzhanskaia**](https://www.linkedin.com/in/anastasiia-derzhanskaia/) (Lead ML Engineer)
- Main tasks: Model development

[**Patrick Fitz**](https://www.linkedin.com/in/patrick-fitz-b2186a11b/) (Lead ML Engineer)
- Main tasks: App development
</td>
</tr>
</table>

**Task Contributors**
<table>
<tr>
</tr>
<tr>
<td>

[**Michael Spurgeon Jr**](https://www.linkedin.com/in/michael-spurgeon-jr-ab3661321/) (ML Engineer)
- Tasks: Data Scraping

[**Pooja Prasad**](https://github.com/prajinip13) (Junior ML Engineer)
- Tasks: Data Review & Cleaning
</td>
<td>

[**Wafa Basudan**](https://www.linkedin.com/in/wafa-basoodan-9447616a/) (ML Engineer)
- Tasks: Model Development

[**Kunal Sood**](https://www.linkedin.com/in/kunal-sood-3b993225b/) (Junior ML Engineer)
- Tasks: Model Development
</td>
<td>

[**Ethel Phiri**](https://www.linkedin.com/in/ethel-phiri261312/) (Junior ML Engineer)
- Main tasks: Data Scraping

[**Gulzar Helvaci**](https://github.com/guluzar-gb) (Junior ML Engineer)
- Main tasks: Data Review & Cleaning
</td>
</tr>
</table>

&nbsp;&nbsp;&nbsp;[**Damla Helvaci**](https://www.linkedin.com/in/damla-helvaci/) (Junior ML Engineer)
- Tasks: Data Review & Cleaning
------------------

[**Michael Spurgeon Jr**](https://www.linkedin.com/in/michael-spurgeon-jr-ab3661321/) (ML Engineer)
- Main tasks: Data Scraping

[**Wafa Basudan**](https://www.linkedin.com/in/wafa-basoodan-9447616a/) (ML Engineer)
- Main tasks: Model Development

[Ethel Phiri](https://www.linkedin.com/in/ethel-phiri261312/) (Junior ML Engineer)
- Main tasks: Data Scraping

[Kunal Sood](https://www.linkedin.com/in/kunal-sood-3b993225b/) (Junior ML Engineer)
- Main tasks: Model development

[Pooja Prasad](https://github.com/prajinip13) (Junior ML Engineer)
- Main tasks: Data Review & Cleaning

[Gulzar Helvaci](https://github.com/guluzar-gb) (Junior ML Engineer)
- Main tasks: Data Review & Cleaning

[Damla Helvaci](https://www.linkedin.com/in/damla-helvaci/) (Junior ML Engineer)
- Main tasks: Data Review & Cleaning


# Report

## Introduction
### Problem statement
*Similar to what we could find in the [Omdenas website](https://www.omdena.com/chapter-challenges/ai-assisted-sign-language-translation-for-brazilian-healthcare-settings)*

The problem systems we started the project with is here.

The scope of this is quite broad and ambitious for a 3-month project. While still aiming high, we kept this in mind to inform our decision making. For example, when deciding which words in the dataset to focus on, we included words that would be most likely to be used in a medical context. And we worked with the thinking that this is an initial proof-of-concept, of a solution we would like to develop further in future.

### Sign language processing
*We could do a brief description of the different tasks that are possible with SLP*
*Also, reference the site we used during research phase*

### Research steps
*Cite the paper we used as reference*

### Plan
We will try to classify videos of LIBRAS words
So not images, and not alphabet signs
If possible, we will focus on healthcare related words, since that is the original goal
But since this is just a PoC, if the data is limited in quantity or quality, we can just focus on general words
We will follow a similar methodology to:
2023 paper collecting public LIBRAS datasets & applying SLT
Key details and points:
A Cross-Dataset Study on the Brazilian Sign Language Translation (2023)
Omdena Indonesia’s Sign Language Translation Project
Key details and points:
Omdena Indonesia - Report Summary
To understand our goal in this project, it would be helpful to read about them.
2023 paper - A Cross-Dataset Study
We will try to use the same 4 source datasets as them
We will try to also explore both CNN & Landmark Based feature extraction methods
Omdena Indonesia’s Sign Language Translation Project
We might record our own videos like they did
We will try to also explore both CNN & Landmark Based feature extraction methods
We will try to create a similar Report & Demo App 

## Data

### Scraping

To build a robust dataset for Brazilian Sign Language (Libras), we identified four sources containing hundreds of videos representing various signs. Due to the scale and structure of these sources, we implemented web automation tools—primarily **Selenium**—to efficiently extract video URLs and relevant metadata.

The scraped data was compiled into a CSV file, which served as a foundational resource for later stages of data cleaning, preprocessing, and modeling.

**Data Sources:**

* [INES](https://www.ines.gov.br/dicionario-de-libras/)
* [Signbank (UFSC)](https://signbank.libras.ufsc.br/pt)
* [UFV Dictionary](https://sistemas.cead.ufv.br/capes/dicionario/)
* [V-Librasil Dataset](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras#files)


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

With the extracted videos, a cleaning step was applied. This task should solve and help with a few issues:

1. **Homographs:** Like any other language, there are some cases of homographs in Portuguese, that is, words that have the same spelling and different meanings, an example of homograph in English is the word "bat" that could either refer to the animal or the object used to hit the ball in a baseball game. Due to the structure of the data, these words would be registered with the same label but have different signs.
2. **Sign sinonyms:** Some words could have more than one sign, depending on the region or signer. Similarly to the homographs, these words would be registred under the same label but have different signs in the videos.
3. **Lack of video examples:** Even though there were videos for every registered word in the datasets, some of them had fewer videos than others. For the project to be successful, it was necessary to have at least a certain amount of example videos to train the models.
4. **Healthcare related words:** The project's main goal was to act on translating LIBRAS specifically in healthcare settings. Therefore, it was necessary to select health related words to compose a target dataset.

Below, you can find the approach used to solve each of the issues above.

#### Video reviewing
[comment]: <TODO: continue video reviewing and data cleaning steps from here>


### Review steps

### Final dataset


## Preprocessing

![alt text](assets/placeholder.png "placeholder")


<img src="assets/placeholder.png" alt="isolated" width="200"/>

<img src="assets/placeholder.png" alt="isolated" title="hover hint" style="width: 75%;"/>

### EDA

- Dataset differences
    - dimensions
        - grid of data sources & dims
    - frame rates
        - on same grid as above?
    - durations
        - plot for each label

#### Dimensions & FPS

#### Durations

### Pose estimation with [MediaPipe Holistic](https://ai.google.dev/edge/mediapipe/solutions/guide)
why:
    - will be used in preprocessing for motion detection, offset & scaling
    - Will be used as the base features that will be input to the model

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


## Model development
### Landmark -> LSTM method
#### Overview
#### Train / Validation / Test split
#### Feature Engineering
#### Data Augmentation
#### Models

## Results
### Overview
### Analysis
### Future ideas

## Demo app