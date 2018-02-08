from flask import Flask, url_for, send_from_directory, request, render_template
import logging, os
import torch, torchvision
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


MODEL_FILE_NAME = 'pretrained/latest_net_G.pth'

model = torch.load(MODEL_FILE_NAME)
model.cpu()
model.train(False)
model.eval()

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

valid_transform = torchvision.transforms.Compose([
    #torchvision.transforms.Resize(size=256, interpolation=Image.ANTIALIAS),
    torchvision.transforms.ToTensor()
])

counter = 0
size = 256

@app.route('/')
def startup():
        return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    global counter
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']

        if os.path.isfile("tmp/output" + str(counter) +".png"):
            os.remove("tmp/output" + str(counter) +".png")
        img = image_process(img)
        img_tensor = predict(img)
        img_numpy = tensor2im(img_tensor)
        numpy_img = Image.fromarray(img_numpy)

        counter = uniqid()

        numpy_img.save("tmp/output" + str(counter) + ".png" , "PNG")
        remove_background("tmp/output" + str(counter) + ".png")
        filename = "output" + str(counter) + ".png"

        return render_template('index2.html', image_name=filename)
    else:
        print("where is the image?")


def predict(im):
    input_batch = []
    im = im.convert('RGB')
    input_batch.append(valid_transform(im))
    #print(input_batch)
    input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
    #print(input_batch_var)
    return model(input_batch_var)


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].data.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 *255.0
    return image_numpy.astype(imtype)


def image_process(img):
    global size
    im = Image.open(img)
    im.thumbnail([size, size], Image.ANTIALIAS)
    cropped = ImageOps.fit(im, (size, size), Image.ANTIALIAS, (0.5, 0.5))
    return cropped


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

def uniqid():
    from time import time
    return hex(int(time()*10000000))[2:]



@app.route('/Flask_API/<filename>')
def send_image(filename):
    return send_from_directory("tmp/", filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

