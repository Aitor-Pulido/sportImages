from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import keras

sport_model = keras.models.load_model('sports_mnist.h5py')

images=[]
# AQUI ESPECIFICAMOS UNAS IMAGENES
filenames = ['1.jpg','basket.jpg','golf.jpg','balon.jpg','pelota_golf.jpg']
# Por algun motivo solo acepta jpg, aunque bake.png es de la pagina del codigo
for filepath in filenames:
    image = plt.imread(filepath,0)
    image_resized = resize(image, (21, 28),anti_aliasing=True,clip=False,preserve_range=True)
    images.append(image_resized)

X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
test_X = X.astype('float32')
test_X = test_X / 255.

predicted_classes = sport_model.predict(test_X)



imgpath = "C:\\Users\\Carlos\\Downloads\\sportimages"
#imgpath = "C:\\Users\\Carlos\\Downloads\\Resultados"
directories = []
prevRoot=''
cant=0

for root, dirnames, filenames2 in os.walk(imgpath):
    for filename in filenames2:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            if prevRoot !=root:
                prevRoot=root
                directories.append(root)
                cant=0

deportes=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    deportes.append(name[len(name)-1])
    indice=indice+1


print(predicted_classes)
print(deportes)

for i, img_tagged in enumerate(predicted_classes):
    print(filenames[i], deportes[img_tagged.tolist().index(max(img_tagged))])