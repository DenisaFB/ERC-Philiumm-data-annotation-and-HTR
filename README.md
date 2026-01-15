This repository contains 1 manually annotated dataset used for the fine-tuning of segmentation models and 2 different tests that we did for transcribing text.

Step 1 : Annotating data in order to fine-tune a segmentation model capable of recognizing the main zones of text, margin text, figures, source quotation zone.

Note : We decided not to crop the images, when we have two pages in one image, because we noticed that sometimes lines started on the left page continue on the right page. Instead, we tought our model to learn how to differenciate the text in one page from the one in another page by annotating the pages into separate zones.

Delimitating text in different zones is important for the next reasons : 

1. **reading order** : the MainZone is the one usually having the most lines of text. We want to separate these lines from the margin text zones or from other MainZone. The reading order is typically from left to right for Latin based text. The transcription will follow this order for the zones and the lines of text.
2. **encoding visual cues in XML-TEI or Word** : we can later extract the content of these zones and encode it in XML-TEI or structure it in a Word text edition.
3. **separating lines** : if we have uniquely one zone, the line detection model will most likely detect one line spanning over multiple zones (following the rule of reading from left to right) : this can impact the quality of the HTR as we cannot make sense of the text if we extract it, since the logical order of lines is not followed.
4. **alignment** : we want to improve the existing HTR models and we have editions in PDF format that we could align with the result of the current HTR best models. If the lines are correctly detected, we have better chances in creating more ground truth data in a more rapid way.

**dataset_RF-DETR-Seg-Preview_model** : Dataset containing the annotation of 280 annotated images containing little or no mathematical expressions + 58 images containing a lot of mathematical expressions annotated by Yunfan.

Note : for the mathematical expression images, we only corrected the following zones : MainZone, MarginTextZone, GraphicZone-figure.

Initial classes in the mathematical expression dataset annotated by Yunfan : 
1. figure -> GraphicZone-figure
2. math -> GraphicZone-formula-inline
3. mathbarree -> GraphicZone-formula-strikethrough
4. mathstructuree -> GraphicZone-formula-complex
5. texte -> MainZone
6. texteMath -> éliminé

Final Classes in this dataset : 
1. DigitizationArtefactZone : 194
2. GraphicZone-figure : 407
3. GraphicZone-formula-complex	: 123
4. GraphicZone-formula-inline	: 1 430
5. GraphicZone-formula-strikethrough	: 856
6. MainZone : 509
7. MarginTextZone : 590
8. NumberingZone	: 310

Our best model for zone segmentation with mathematical expression samples is the following : https://app.roboflow.com/ercphiliumm/layout-detection-leibniz-manuscripts-6wvnb/models/layout-detection-leibniz-manuscripts-6wvnb/23

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

Our best model for zone segmentation without mathematical expression samples is the following : https://app.roboflow.com/ercphiliumm/layout-detection-leibniz-manuscripts-6wvnb/models/layout-detection-leibniz-manuscripts-6wvnb/25

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

Observations : 
- This model performs well on real data. We will continue to add new data as we correct on the eScriptorium interface during the ground truth production and try to improve it. The model is having less certainty for the MarginTextZone class (it is expected, sometimes we have these uncertainties ourselves during the annotation process).
- We noticed that Instance Segmentation models do not perform well with overlapping zones. It will generally prefer the larger zones to the smaller ones. This explains why the mathematical expression zones are often overlooked, especially if they are situated inside a MainZone or a MarginTextZone.
- We tested YOLO (v5 and v11) models and they are less performant on our data than the RF-DETR model.
- We tested Detectron 2 instance segmentation model (not yet on mathematical samples) and the results are promising. We did not find the optimal parameters for this model yet.
- Detectron 2 proposes panoptic segmentation as well. This could possibly solve the issue of overlapping zones and converge maybe two different models (one for the large zones and one for the mathematical expressions) into one model. We did not test it yet, since we need to see how to learn the data.

Step 2 : HTR on the predicted zones.
