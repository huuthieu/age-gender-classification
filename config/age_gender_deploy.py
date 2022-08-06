# import the necessary packages
from config.age_gender_config import OUTPUT_BASE
from os import path
import os

ROOT = '/content/age-gender-classification/'
# define the path to the dlib facial landmark predictor
DLIB_LANDMARK_PATH = ROOT+ "shape_predictor_68_face_landmarks.dat"

# define the path to the age network + supporting files
AGE_NETWORK_PATH = ROOT+ "checkpoints/age"
AGE_PREFIX = "agenet"
AGE_EPOCH = 109
AGE_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, "age_le.cpickle"])
AGE_MEANS = path.sep.join([OUTPUT_BASE, "age_adience_mean.json"])

# define the path to the gender network + supporting files
GENDER_NETWORK_PATH = ROOT+ "checkpoints/gender"
GENDER_PREFIX = "gendernet"
GENDER_EPOCH = 110
GENDER_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, "gender_le.cpickle"])

GENDER_MEANS = path.sep.join([OUTPUT_BASE, "gender_adience_mean.json"])

