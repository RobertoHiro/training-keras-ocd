import keras_ocr
import matplotlib.pyplot as plt

pipeline = keras_ocr.pipeline.Pipeline()

images = [
    keras_ocr.tools.read(img) for img in ['/content/BoletoBancario.png',]
]

# generate text predictions from the images
prediction_groups = pipeline.recognize(images)

# plot the text predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, 
                                    predictions=predictions, 
                                    ax=ax)

predicted_image = prediction_groups[1]
for text, box in predicted_image:
    print(text)