# Text and Formula Zone Detection and Transcription: Initial Experiments

This repository contains 1 manually annotated dataset used for the fine-tuning of different segmentation models and 2 different tests that we did for transcribing the text inside the predicted zones.

## Step 1 : Annotating data in order to fine-tune a segmentation model capable of recognizing the main zones of text, margin text, figures, source quotation zone.

Note : We decided not to crop the images, when we have two pages in one image, because we noticed that sometimes lines start on the left page and continue on the right page. Instead, we tought our model to learn how to differenciate the text in one page from the one in another page by annotating these pages into separate zones.

Delimitating text in different zones is important for the following reasons : 

1. **reading order** : the MainZone is the one usually having the most lines of text. We want to separate these lines from the margin text zones or from other main zones. The reading order is typically from left to right for Latin based text. The transcription will follow this rule for the zones and for the lines of text.
2. **encoding visual cues in XML-TEI or Word** : we can later extract the content of these zones and encode it in XML-TEI or structure it in a Word text edition.
3. **separating lines** : if we have one zone, the line detection model will most likely detect one line spanning over multiple logical zones (following the rule of reading from left to right). This can impact the quality of the HTR as we cannot make sense of the text if we extract it, since the logical order of lines is not followed.
4. **alignment** : we want to improve the existing HTR models and we have editions in PDF format that we could align with the result of the current HTR best models. If the lines are correctly detected, we have better chances in creating more ground truth data in a more rapid way.

Folder 1 : **dataset_RF-DETR-Seg-Preview_model** : Dataset containing 280 annotated images with little or no mathematical formulas + 58 images with a lot of mathematical formulas annotated by Yunfan.

⚠️ These annotations represent the last corrected version - this is not the version of the dataset we used for the fine-tuning of the two models quoted below (we used different versions, depending at which stage we were in the annotation process).

**Annotations format** : COCO Segmentation (see "_annotations.coco.json" files in train/valid/test sets)

Note : for the images from Yunfan's dataset, we only corrected the following zones : MainZone, MarginTextZone, GraphicZone-figure.

We used the SegmOnto standard for our classes : https://segmonto.github.io/.

The SegmOnto standard does not propose specific classes for mathematical formula yet, but we used the GraphicZone type and assigned the following attributes : formula-complex, formula-inline, formula-strikethrough after analyzing the annotations.

We can add any attribute that is useful in describing our zones. This standard is highly used in annotating zones prior to the HTR/OCR step.

Initial classes in the mathematical formula dataset annotated by Yunfan and the classes we assigned based on the SegmOnto standard : 
1. figure -> GraphicZone-figure
2. math -> GraphicZone-formula-inline
3. mathbarree -> GraphicZone-formula-strikethrough
4. mathstructuree -> GraphicZone-formula-complex
5. texte -> MainZone
6. texteMath -> eliminated

Final Classes in this dataset : 
1. DigitizationArtefactZone : 194
2. GraphicZone-figure : 407
3. GraphicZone-formula-complex	: 123
4. GraphicZone-formula-inline	: 1 430
5. GraphicZone-formula-strikethrough	: 856
6. MainZone : 509
7. MarginTextZone : 590
8. NumberingZone	: 310

Our best model for zone segmentation with mathematical formula samples is the following : https://app.roboflow.com/ercphiliumm/layout-detection-leibniz-manuscripts-6wvnb/models/layout-detection-leibniz-manuscripts-6wvnb/23

Model type : instance segmentation model 
Pre-trained model : **RF-DETR-Seg-Preview**

- mAP@50 : 57.6%
- Precision : 61.4%
- Recall : 60.0%

Class Evaluation on the validation set : 
- DigitizationArtefactZone : 97.0%
- GraphicZone-figure : 84.0%
- GraphicZone-formula : 38.0%
- GraphicZone-formula-complex : 25.0%
- GraphicZone-formula-inline : 13.0%
- GraphicZone-formula-strikethrough : 12.0%
- MainZone : 97.0%
- MarginTextZone : 74.0%
- NumberingZone : 78.0%
- all : 58.0%

Class Evaluation on the test set : 
- DigitizationArtefactZone : 100%
- GraphicZone-figure : 75.0%
- GraphicZone-formula : 1.0%
- GraphicZone-formula-complex :1.0%
- GraphicZone-formula-inline : 20.0%
- GraphicZone-formula-strikethrough : 7.0%
- MainZone : 100%
- MarginTextZone : 88.0%
- NumberingZone : 81.0%
- all : 52.0%

Our best model for zone segmentation without mathematical formula samples is the following : https://app.roboflow.com/ercphiliumm/layout-detection-leibniz-manuscripts-6wvnb/models/layout-detection-leibniz-manuscripts-6wvnb/25

Model type : instance segmentation model 
Pre-trained model : **RF-DETR-Seg-Preview**

- mAP@50 : 88.6%
- Precision : 88.4%
- Recall : 79.0%

Class Evaluation on the validation set : 
- DigitizationArtefactZone : 100%
- GraphicZone-figure : 78.0%
- MainZone : 99.0%
- MarginTextZone :78.0%
- NumberingZone : 88.0%
- all : 89.0%

Class Evaluation on the test set : 
- DigitizationArtefactZone : 100%
- GraphicZone-figure : 72.0%
- MainZone : 100%
- MarginTextZone : 91.0%
- NumberingZone : 94.0%
- all : 91.0%

**Observations :**
- This model performs well on real data. We will continue to add new data as we correct on the eScriptorium interface during the ground truth production and try to improve it. The model is having less certainty for the MarginTextZone class (it is expected, sometimes we have these uncertainties ourselves during the annotation process).
- We noticed that Instance Segmentation models do not perform well with overlapping zones. It will generally prefer the larger zones to the smaller ones. This explains why the mathematical formula zones are often overlooked, especially if they are situated inside a MainZone or a MarginTextZone.

**Other segmentation models tested :**
- We tested YOLO (v8 and v11 - these versions propose instance segmentation) models and they are less performant on our data than the RF-DETR model.
- We tested Detectron 2 instance segmentation model (not yet on mathematical samples) and the results are promising. We did not find the optimal parameters for this model yet.
- Detectron 2 proposes panoptic segmentation as well. This could possibly solve the issue of overlapping zones and converge maybe two different models (one for the large zones and one for the mathematical formulas) into one model. We did not test it yet, since we need to learn how to prepare the data.

**Tools used :**
- Google Colab for YOLO and Detectron 2 model fine-tuning.
- Roboflow API for data annotation and RF-DETR model fine-tuning.

## Step 2 : HTR on all of the predicted zones with and without mathematics samples.

Folder 2 : **HTR_all**

This folder contains all images of the annotated dataset and their corresponding HTR results in three formats : XML Page, XML ALTO and TXT. The XML files contain all coordinated of all TextRegions (zones) and of all TextLines (the lines of text).

For this experience we did not ignore the mathematical formula zones in the transcription.

Pre-processing : 
- mask out GraphicZone type zones prior to the line detection.
- clip lines that span from one zone to another.

Issues we noticed : 

The MainZone and the different mathematical formula zones are overlapping. Hence, in the HTR result (see XML Page or XML Alto files) the lines inside the mathematical zones are assigned all to the MainZone.

We could solve this by creating "holes" in the MainZone, dedicated to the mathematical formula zones. This would create separate zones. This made us wonder if we should cut the math zones out of the larger zones before fine-tuning a segmentation model. We are not sure how this would affect the overall performance of the model on pages that do not have mathematical formulas for exemple.

## Step 3 : HTR focused solely on the mathematical formula samples - without transcribing text inside the mathematics zones.

Folder 3 : **HTR_excluding_math_zones**

Pre-processing : 
- mask out zones that are mathematical zones and GraphicZone type zones prior to the line detection : see subfolder "Zone_masks". The black pixels are the only ones allowed for transcription.

Issues we noticed : 

Sometimes, mathematical formulas are followed or preceded by small segments of text. In the end we would have one line containing the text + the LATEX transcription of the mathematical formula. At the moment, since we ignored mathematical formula transcription, we have one line fragmented into many small line segments.

We used the tool eScriptorium to visualise and correct the zones and lines detected and to run the HTR model : https://gitlab.com/scripta/escriptorium.

--------
Author : Denisa-Florina Bumba, ERC Philiumm Project
