#encoding:utf-8
from PIL import Image
import numpy as np
import sys
if __name__ == '__main__':
    # 读取图片转成灰度格式
    img_name = 'images/'+sys.argv[1]
    img = Image.open(img_name).convert('L')

    # resize的过程
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28),Image.ANTIALIAS)
    # print(np.array(img))


    # 暂存像素值的一维数组
    arr = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            # mnist 里的颜色是0代表白色（背景），1.0代表黑色
            # pixel = 1.0 - float(img.getpixel((j, i)))/255.0
            pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
            arr[i,j] = pixel
    # print(arr)
    img = Image.fromarray(np.uint8(arr))
    # img.show()
    img.save('mnist/'+sys.argv[1])


