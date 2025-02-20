# Image Captioning
## What is Image Captioning ?

Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions.
This task lies at the intersection of computer vision and natural language processing. Most image captioning systems use an encoder-decoder framework, where an input image is encoded into an intermediate representation of the information in the image, and then decoded into a descriptive text sequence.
CNNs + RNNs (LSTMs)

To perform Image Captioning we will require two deep learning models combined into one for the training purpose
CNNs extract the features from the image of some vector size aka the vector embeddings. The size of these embeddings depend on the type of pretrained network being used for the feature extraction
LSTMs are used for the text generation process. The image embeddings are concatenated with the word embeddings and passed to the LSTM to generate the next word
For a more illustrative explanation of this architecture check the Modelling section for a picture representation

![imagecaption](https://github.com/user-attachments/assets/98cd947f-833f-4381-9d12-3c8d12d81f5c)
# Caption Text Preprocessing Steps
Convert sentences into lowercase
Remove special characters and numbers present in the text
Remove extra spaces
Remove single characters
Add a starting and an ending tag to the sentences to indicate the beginning and the ending of a sentence
# Tokenization and Encoded Representation
The words in a sentence are separated/tokenized and encoded in a one hot representation
These encodings are then passed to the embeddings layer to generate word embeddings
![68747470733a2f2f6c656e612d766f6974612e6769746875622e696f2f7265736f75726365732f6c656374757265732f776f72645f656d622f6c6f6f6b75705f7461626c652e676966](https://github.com/user-attachments/assets/1597f543-10f7-4ba5-93f9-88b099568ca2)

# Image Feature Extraction
DenseNet 201 Architecture is used to extract the features from the images
Any other pretrained architecture can also be used for extracting features from these images
Since the Global Average Pooling layer is selected as the final layer of the DenseNet201 model for our feature extraction, our image embeddings will be a vector of size 1920

# Data Generation
Since Image Caption model training like any other neural network training is a highly resource utillizing process we cannot load the data into the main memory all at once, and hence we need to generate the data in the required format batch wise
The inputs will be the image embeddings and their corresonding caption text embeddings for the training process
The text embeddings are passed word by word for the caption generation during inference time

# Modelling
The image embedding representations are concatenated with the first word of sentence ie. starseq and passed to the LSTM network
The LSTM network starts generating words after each input thus forming a sentence at the end

# Caption Generation Utility Functions
Utility functions to generate the captions of input images at the inference time.
Here the image embeddings are passed along with the first word, followed by which the text embedding of each new word is passed to generate the next word
![canva](https://github.com/user-attachments/assets/620ca5d2-85b7-4953-ac63-6193b1d875bc)

