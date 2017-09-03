from utilities import *
import networks
''' 1
img = "../shanghai/shanghai_1.png"

i = Image.open(img)
t = image_to_tensor(i, i.size)
## i.show()
print(t.size())
'''
''' 2
image_dir = "../shanghai"
label_file = "../shanghai_points_list.csv"
with open(label_file, "r") as f1:
	with open("../labels.csv", "w+") as f2:
		for row in f1:
			num, lab = row.split(",")
			f2.write("shanghai_%s.png,%s" % (num, lab))
'''

input_dir = "../shanghai"
label_file = "../labels.csv"
images_list, labels = image_label(input_dir, label_file)

images_dataset = ImageDataset(images_list, labels, image_size=(224, 224))

vgg16 = networks.vgg16_pretrained

imgt = Variable(images_dataset[0][0])
output = vgg16(imgt)
print(output.data.numpy().shape)