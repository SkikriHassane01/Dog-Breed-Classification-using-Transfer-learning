import numpy as np
import streamlit as st
import cv2

# load the keras model 
from tensorflow.keras.models import load_model
model = load_model('dogBreedModel.kear')

# How many class are there?
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 120
CLASS_NAMES = ['boston_bull', 'dingo', 'pekinese', 'bluetick', 'golden_retriever',
       'bedlington_terrier', 'borzoi', 'basenji', 'scottish_deerhound',
       'shetland_sheepdog', 'walker_hound', 'maltese_dog',
       'norfolk_terrier', 'african_hunting_dog',
       'wire-haired_fox_terrier', 'redbone', 'lakeland_terrier', 'boxer',
       'doberman', 'otterhound', 'standard_schnauzer',
       'irish_water_spaniel', 'black-and-tan_coonhound', 'cairn',
       'affenpinscher', 'labrador_retriever', 'ibizan_hound',
       'english_setter', 'weimaraner', 'giant_schnauzer', 'groenendael',
       'dhole', 'toy_poodle', 'border_terrier', 'tibetan_terrier',
       'norwegian_elkhound', 'shih-tzu', 'irish_terrier', 'kuvasz',
       'german_shepherd', 'greater_swiss_mountain_dog', 'basset',
       'australian_terrier', 'schipperke', 'rhodesian_ridgeback',
       'irish_setter', 'appenzeller', 'bloodhound', 'samoyed',
       'miniature_schnauzer', 'brittany_spaniel', 'kelpie', 'papillon',
       'border_collie', 'entlebucher', 'collie', 'malamute',
       'welsh_springer_spaniel', 'chihuahua', 'saluki', 'pug', 'malinois',
       'komondor', 'airedale', 'leonberg', 'mexican_hairless',
       'bull_mastiff', 'bernese_mountain_dog',
       'american_staffordshire_terrier', 'lhasa', 'cardigan',
       'italian_greyhound', 'clumber', 'scotch_terrier', 'afghan_hound',
       'old_english_sheepdog', 'saint_bernard', 'miniature_pinscher',
       'eskimo_dog', 'irish_wolfhound', 'brabancon_griffon',
       'toy_terrier', 'chow', 'flat-coated_retriever', 'norwich_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'english_foxhound', 'gordon_setter', 'siberian_husky',
       'newfoundland', 'briard', 'chesapeake_bay_retriever',
       'dandie_dinmont', 'great_pyrenees', 'beagle', 'vizsla',
       'west_highland_white_terrier', 'kerry_blue_terrier', 'whippet',
       'sealyham_terrier', 'standard_poodle', 'keeshond',
       'japanese_spaniel', 'miniature_poodle', 'pomeranian',
       'curly-coated_retriever', 'yorkshire_terrier', 'pembroke',
       'great_dane', 'blenheim_spaniel', 'silky_terrier',
       'sussex_spaniel', 'german_short-haired_pointer', 'french_bulldog',
       'bouvier_des_flandres', 'tibetan_mastiff', 'english_springer',
       'cocker_spaniel', 'rottweiler']
# Title of the app
st.title('Dog Breed Classifier')
st.subheader('Upload a picture of a dog and we will tell you the breed')

# Upload the dog image
dog_image = st.file_uploader('Upload the image of the dog', type=['jpg', 'png'])

# prediction button
submit = st.button('Predict')

# Display the prediction
if submit:

    if dog_image is not None:

        # 1.convert the file to an opencv image
        open_cv_image = np.array(bytearray(dog_image.read()), dtype=np.uint8)
        # imdecode is used to convert the image into a format that opencv can understand
        img = cv2.imdecode(open_cv_image, 1) # 1 means load a color image

        # Display the image
        st.image(img, channels='BGR')

        # 2.convert the image to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Resize the image 
        img = cv2.resize(img, IMAGE_SIZE)

        # 4. Make the prediction
        prediction = model.predict(np.array([img]))

        # 5. Get the class name
        class_name = CLASS_NAMES[np.argmax(prediction)]

        # 6. Display the class name
        st.title(f'The dog breed is {class_name}')


