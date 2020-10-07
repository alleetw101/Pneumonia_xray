# pneumonia_xray

Author: Alan Lee <alleetw101@gmail.com>  
*Last updated: 2020.10*

## Description

An image classification project to identify pediatric chest x-ray images with pneumonia. 

From the National Institute of Health (NIH):
> Pneumonia is an infection that affects one or both lungs. It causes the air sacs, or alveoli, of the lungs to fill 
> up with fluid or pus. Bacteria, viruses, or fungi may cause pneumonia. Symptoms can range from mild to serious 
> and may include a cough with or without mucus (a slimy substance), fever, chills, and trouble breathing. (2020.10.06)

## Purpose

This is my first full machine learning (ML) project. It is/was intended for learning purposes and to increase 
familiarity with development tools. Experimentation is the primary goal of the project with performance as secondary. 

Focused Topics:
- Python
- Tensorflow
    - data
    - saved_model
- Keras
- PyCharm IDE
- Git, Github
- Command line (Terminal)

## Dataset

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five 
years old from Guangzhou Women and Children’s Medical Center, Guangzhou. The diagnoses for the images were graded by 
two expert physicians. 

This project utlized the training set (5216 JPEG files) and the testing set (624 JPEG files) at a total size of 1.23 GB. 
Validation set was excluded from project due to its small sample size (16 JPEG files). Images were categorized into 
-NORMAL- and -PNEUMONIA- for each set. No external preprocessing was applied.

Obtained from [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
Original data: https://data.mendeley.com/datasets/rscbjbr9sj/2  
License: CC BY 4.0  
Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## Specifications
### Project

- Python 3.8
- Tensorflow 2.3

### Development

Macbook Pro (15-inch, 2017)
- CPU: 2.9 GHZ (i7-7820HQ)
- GPU: None utilized
- RAM: 16 GB
- OS: macOS 10.15 (Catalina)

PyCharm 2020.2 (CE)

Google Colab
