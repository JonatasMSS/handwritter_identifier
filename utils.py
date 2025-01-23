

from PIL import Image, ImageOps
# Function that save image


def save_image(img_data):
    data = img_data
    data = Image.fromarray(data)
    data = data.convert('L')
    data = ImageOps.invert(data)
    data = data.resize((100, 100))
    data.save('temp_img.png')
