<mark>This page is under construction</mark>

# Healthcare Sign Language Brazil


## Summary
*This should be somewhat similar to what is described in the [project's main page](https://www.omdena.com/chapter-challenges/ai-assisted-sign-language-translation-for-brazilian-healthcare-settings)*

### Goal

### Contributors
Here's a list of people that contributed to the project. Feel free to reach out to them if needed:

[Ben Thompson](https://www.linkedin.com/in/ben-d-thompson/) (Project Manager)

[Ayushya Pare](https://www.linkedin.com) (Data Scraping)

[Gustavo Santos](https://www.linkedin.com/in/gustavopsantos) (Data cleaning)

[Anastasiia Derzhanskaia](https://www.linkedin.com) (Model development)

[Kunal Sood](https://www.linkedin.com) (Model development)

[Patrick Fitz](https://www.linkedin.com) (App development)

[Damla Helvaci](https://www.linkedin.com)

[Ethel Phiri](https://www.linkedin.com/in/ethel-phiri261312/)

[Gulzar](https://www.linkedin.com)

[Michael Spurgeon Jr](https://www.linkedin.com)

[Pooja Prasad](https://www.linkedin.com)

[Lorem Ipsum](https://www.linkedin.com) (Main role)


### Results
*Briefly describe main results and use a [link] (#results-1) reference somewhere in your text to send the reader to a complete version of the results* 

### Demo app
*Briefly describe the app and use a [link] (#demo-app-1) reference somewhere in your text to send the reader to a more complete description of the app* 

## Introduction
### Problem statement
*Similar to what we could find in the Omdenas website*

### Sign language processing
*We could do a brief description of the different tasks that are possible with SLP*
*Also, reference the site we used during research phase*

### Research steps
*Cite the paper we used as reference*

## Data
### Scraping
### Cleaning
### Review steps
### Final dataset

## Preprocessing
### EDA
*DRAFT*
- Dataset differences
    - 
*DRAFT*
### Pose estimation with [MediaPipe Holistic](https://ai.google.dev/edge/mediapipe/solutions/guide)
### Start/End point definition
### Scaling and align videos
### Interpolating `none` frames
*DRAFT*
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

*DRAFT*

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