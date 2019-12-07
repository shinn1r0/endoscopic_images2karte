from PIL import Image, ImageFilter


im = Image.open('../images/0007955938/0076_0007955938_201701051036_3b087b95.jpg')
print(im.format, im.size, im.mode)