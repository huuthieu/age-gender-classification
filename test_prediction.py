# import OpenCV before mxnet to avoid a segmentation fault
import cv2

# import the necessary packages
from config import age_gender_deploy as deploy
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.utils.agegenderhelper import AgeGenderHelper
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import dlib
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image (or directory)")
args = vars(ap.parse_args())

# load the label encoders and mean files
print("[INFO] loading label encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
ageMeans = json.loads(open(deploy.AGE_MEANS).read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

# load the models from disk
print("[INFO] loading models...")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,
    deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,
    deploy.GENDER_PREFIX])
ageModel = mx.model.FeedForward.load(agePath, deploy.AGE_EPOCH)
genderModel = mx.model.FeedForward.load(genderPath,
    deploy.GENDER_EPOCH)

# now that the networks are loaded, we need to compile them
print("[INFO] compiling models...")
ageModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
    symbol=ageModel.symbol, arg_params=ageModel.arg_params,
    aux_params=ageModel.aux_params)
genderModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
    symbol=genderModel.symbol, arg_params=genderModel.arg_params,
    aux_params=genderModel.aux_params)


# initialize the image pre-processors
sp = SimplePreprocessor(width=256, height=256,  
    inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227, height=227, horiz=True)
ageMP = MeanPreprocessor(ageMeans["R"], ageMeans["G"],
    ageMeans["B"])
genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"],
    genderMeans["B"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

# initialize dlib’s face detector (HOG-based), then create the
# the facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# initialize the list of image paths as just a single image
imagePaths = [args["image"]]

# if the input path is actually a directory, then list all image
# paths in the directory
if os.path.isdir(args["image"]):
    imagePaths = sorted(list(paths.list_files(args["image"])))

# loop over the image paths
for imagePath in imagePaths:
    # load the image from disk, resize it, and convert it to
    # grayscale
    print("[INFO] processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # align the face
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)

        # resize the face to a fixed size, then extract 10-crop
        # patches from it
        face = sp.preprocess(face)
        patches = cp.preprocess(face)

        # allocate memory for the age and gender patches
        agePatches = np.zeros((patches.shape[0], 3, 227, 227),
            dtype="float")
        genderPatches = np.zeros((patches.shape[0], 3, 227, 227),
            dtype="float")

        # loop over the patches
        for j in np.arange(0, patches.shape[0]):
            # perform mean subtraction on the patch
            agePatch = ageMP.preprocess(patches[j])
            genderPatch = genderMP.preprocess(patches[j])
            agePatch = iap.preprocess(agePatch)
            genderPatch = iap.preprocess(genderPatch)

            # update the respective patches lists
            agePatches[j] = agePatch
            genderPatches[j] = genderPatch
        
            # make predictions on age and gender based on the extracted
            # patches
            agePreds = ageModel.predict(agePatches)
            genderPreds = genderModel.predict(genderPatches)

            # compute the average for each class label based on the
            # predictions for the patches
            agePreds = agePreds.mean(axis=0)
            genderPreds = genderPreds.mean(axis=0)   
            # visualize the age and gender predictions
            ageCanvas = AgeGenderHelper.visualizeAge(agePreds, ageLE)
            genderCanvas = AgeGenderHelper.visualizeGender(genderPreds,
                genderLE)

            img_name = os.path.basename(imagePath)

            # draw the bounding box around the face
            clone = image.copy()
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("{}results/{}_Input.jpg".format(deploy.ROOT, img_name), clone)
            cv2.imwrite("{}results/{}_Face.jpg".format(deploy.ROOT, img_name), face)
            cv2.imwrite("{}results/{}_AgeProbabilities.jpg".format(deploy.ROOT, img_name), ageCanvas)
            cv2.imwrite("{}results/{}_GenderProbabilities.jpg".format(deploy.ROOT, img_name), genderCanvas)