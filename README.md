# WordAttributeClassification

Classification of Word font attributes in the classes : Bold , italic and Underlined.
## Authors

[Rohan Kumar](https://github.com/codewithrk-108)

## Progress

1. **Week 1**

     * Reading about Tesseract : [Tesseract Blog](https://nanonets.com/blog/ocr-with-tesseract/)
     * Collection of the Research papers on Word attribute Classifications : [List of all Papers](https://docs.google.com/spreadsheets/d/1Cpr64s3GwOkqt_BC3w2SKsfgaAF4JuAtSA3XqibjtLc/edit#gid=0)
     * Checking out TesserOCR, Python Wrapper for Tesseract function
     * Found issues in the Tesseract Classification : [Github Issues](https://github.com/sirfz/tesserocr/issues/292)

```python
"""
The function I was searching for my project.
"""
# e is the iterator at the word level
for e in tesserocr.iterate_level(iterator,tesserocr.RIL.WORD):
    attributes = e.WordFontAttributes()
"""
Here attributes is the dictionary containing :
- Font name
- Bold
- Italic
- Underlined
- Serif
- Font size
"""
```

2. **Week 2**

      * Found an amazing Blog on the levels of Page layout (Tesseract) : [Blog](https://medium.com/geekculture/tesseract-ocr-understanding-the-contents-of-documents-beyond-their-text-a98704b7c655)
      * Found a loop hole for less accuracy on Tesseract classifier using word images: [PSM mode blog](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)
      * Image Processing and OpenCV library (checked out some basics) : [Resources](https://homepages.inf.ed.ac.uk/rbf/HIPR2/index.htm)
      * Built a pipeline for the tesseract word font classification with desired PSM mode.
[Week 2](https://github.com/codewithrk-108/WordAttributeClassification/tree/master/Week-2)

2. **Week 3**

      * Went through the Research Papers collected in Week 1.(More details in PPT)
      * Main Paper Read : CONSENT : Context aware Transformer based Bold Words Classification
      [CVPR 2022 Paper](https://arxiv.org/pdf/2205.07683.pdf)
      *  Read about the Gentle Introduction to Transformers (Attention Mechanisms) in regard to the paper. [Best Blog](https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-2-bf2403804ada),
      [Best Video](https://www.youtube.com/playlist?list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb)

3. **Week 4**

      * Understood the skeletonization and Medial axis transform : [Docs](https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html), [Week 4](https://github.com/codewithrk-108/WordAttributeClassification/tree/master/Week-4)
      * Character segmentation using inbuilt function in wrapper TesserOCR : [Idea Source](https://stackoverflow.com/questions/43389374/using-tesseract-ocr-for-character-segmentation-only)
      * Implementation of Letter Morphology Voting : [Week 4](https://github.com/codewithrk-108/WordAttributeClassification/tree/master/Week-4)
      * Tried changing the alpha values as the threshold. (Alpha = 0.2)
      * Read RNN, Sequence Modeling, LSTMs 
      and GRUs theoritically and BPTT (mathematically) : [Best Resource](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
      * Evaluation Criterion for Language Models : [Best Resource](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)
  
4. **Week 5 & 6**

      * Found an ADOBE_VFR Dataset on Kaggle for font classification.
      * Checked it to be compatible with my problem statement. (Attributes also given alongside fonts)
      * Downloaded .bcf files and converted to .png files and seperated the labels via python scripts.
      * Thought of doing a Classification based machine learning approach to get through the problem.
      * The method adopted was doing a literature survey of the DeepFont Paper and trying to implement
        it in my domain.

5. **Week 7 & 8** (Based on DeepFont paper)
   
      * Initially found two relatable codebases on github for implementation. [LINK-1](https://github.com/robinreni96/Font_Recognition-DeepFont) [LINK-2](https://github.com/artset/WhatTheFont)
      * Went ahead with the approach for a multilabel classification.
      * Adopted a softmax output layer adding approach initially with 4 outputs. (n,b,i)
      * Maintained a pipeline from training on dataset to testing on the real documents.
      * Initially according to paper, imitated training with feeding 50*50 patches.
      * Initial dataset had around 70000 images with the distribution same as
        the distribution of dataset.
      * Made a prediction dependent assumption for evaluation*** (which was not up to the mark).
      * Tried various techniques like reducing no. of 'none' images in dataset to allow model to
        learn attributes.
      * Gave some results on the real documents (published in slides).

6. **Week 9 & 10** (Based on Custom CNN Arhitecture to adopt to the usecase)
   
      * Changed the approach of 50*50 patches with taking a whole image in 170*1200 frame.
      * The image was in grayscale centered around in the frame.
      * Initially this was done on the fly in the dataloader but the speed was slow and
         the model parameters were also high.
      * Dataset contained only 100 images from each font totalling upto 2,38,300 images.
      * The test accuracy on real data of ADOBE VFR was (4385 images): 60-61 % (also contains flawed)
      * Due to the reason of a bulky model, we took a 56*400 frame instead and tried training the model
         with complete synthetic data.
      * The flaws in the dataset and confusion with attributes in images was identified and
         removed which resulted in removal of around 200 fonts out of 2383 font classes.
      * Now, finally a baseline for the classification was trained on the 21,41,000
         synthetic pure images and gave a real test accuracy of 61 % (appx).
      * Tried fine tuning the model by freezing the Conv layers and increasing learning
         rate by twice on FC layers, but did'nt yeild any results, the accuracy did'nt
         increase but test real data (3791 images) accuracy increased but the unseen real test (914)
         still remained/decreased to 57.22 %. 

 8. **Week 11 & 12**
    T.B.D


## Codes and Implementations
Attached in the this Github Repository.
