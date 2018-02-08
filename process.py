import os
from PIL import Image, ImageOps
size = (256, 256)
input = 'Uploads/'
output = 'processed/'
faced = 'output/images/'

def process_image(image):
    #open image resize and crop to 256*256
    im = Image.open(image)
    im.thumbnail([256, 256], Image.ANTIALIAS)
    cropped = ImageOps.fit(im, size, Image.ANTIALIAS, (0.5, 0.5))

    # create image name variable
    image_file = os.path.basename(image)
    image_name = os.path.splitext(image_file)[0]
    print(image_name)

    #image array
    #append cropped images to array
    images = []
    for i in range(2):
        images.append(cropped)

    #calculations for image combination
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    # create blank image 2*width of single image
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    # combine images into A to B format
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    # create destination path + processed image name
    # same image at destination
    dst_path = os.path.join(output, image_name + "_p" + ".png")
    print(dst_path)
    new_im.save(dst_path)

def p_image_name(image_name):

    name = os.path.splitext(image_name)[0]
    return os.path.join(name + "_p" + "_outputs" + ".png")

def remove_background(image_path):
    upper = 255
    lower = 120
    img = Image.open(image_path)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if lower <= item[0] <= upper and lower <= item[1] <= upper and lower <= item[2] <= upper:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(image_path, "PNG")
    print(image_path)
    print("Image background removed")

#remove_background("/Users/ollieford/Downloads/af02-outputs (1).png")


