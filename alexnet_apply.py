from utilities import *

img = Image.open("cat.jpg")
tens = img2tensor(img)
var = Variable(tens)

alex = models.alexnet(pretrained=True)

output = alex(var)
sorted, indices = torch.sort(output)
print(sorted[:5])
print(indices[:5])