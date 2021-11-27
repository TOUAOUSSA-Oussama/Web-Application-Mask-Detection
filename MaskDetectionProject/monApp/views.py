from django.shortcuts import render
from .detection import detect_and_predict_mask
from tensorflow.keras.models import load_model
import cv2
import imutils
from django.core.files.storage import FileSystemStorage

# Create your views here.

## page de garde :
def index(request):
    # cas ou on possede une image :
    if request.method == 'POST' and request.FILES['fichier'] :
        
        ## Telecharger le modele :
        maskNet = load_model("mask_detector.model")

        # telecharger le modele responsable sur la detection de la face
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        ## Lire l'image  :
        img = request.FILES['fichier']
        fs = FileSystemStorage()
        img_name = fs.save(img.name, img)
        img_url = fs.url(img_name)

        # path_img = fs.url(img_name)
        frame = cv2.imread("media\\" +  img_name)
        frame = imutils.resize(frame, width=400)

        ## Recuperer la localition de chaque face et son prediction
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        ## Afficher les faces detecetes et ses predictions :
        for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        img_url = 'media\\resultat.png'
        cv2.imwrite(img_url, frame)

        # delete the saved file :
        fs.delete(img.name)

        return  render(request, 'resultat.html', {'image_path' : img_url})
        
    # cas inverse :
    else :
        return render(request, "index.html")