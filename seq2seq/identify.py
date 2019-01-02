# -*- coding: utf-8 -*-
import random
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont

w = 30 * 4
h = 60
class Genidentify:


    def randomChar(self):
        '''
        随机生成chr
        :return:返回一个随机生成的chr
        '''
        return chr(random.randint(48, 57))

    def randomBgColor(self):
        '''
        随机生成验证码的背景色
        :return:
        '''
        return (random.randint(50, 100), random.randint(50, 100), random.randint(50, 100))

    def randomTextColor(self):
        '''
        随机生成验证码的文字颜色
        :return:
        '''
        return (random.randint(120, 200), random.randint(120, 200), random.randint(120, 200))


if __name__ == '__main__':

    gen = Genidentify()
    # gen.randomBgColor()
    # gen.randomChar()
    # gen.randomTextColor()

    # 设置字体类型及大小
    font = ImageFont.truetype('arial.ttf', size=36)

    for i in range(500):
        # 创建一张图片，指定图片mode，长宽
        image = Image.new('RGB', (w, h), (255, 255, 255))

        # 创建Draw对象
        draw = ImageDraw.Draw(image)
        # 遍历给图片的每个像素点着色
        for x in range(w):
            for y in range(h):
                draw.point((x, y), fill=gen.randomBgColor())

        # 将随机生成的chr，draw如image
        filename = []
        for t in range(4):
            ch = gen.randomChar()
            filename.append(ch)
            draw.text((30 * t, 10), ch, font=font, fill=gen.randomTextColor())

        # 设置图片模糊
        # image = image.filter(ImageFilter.BLUR)
        # 保存图片
        if i <= 400:
            image.save('/home/ray/datasets/identify/train/{0}.jpg'.format("".join(filename)), 'jpeg')
        elif i > 400 and filename not in os.listdir(r'/home/ray/datasets/identify/train'):
            image.save('/home/ray/datasets/identify/test/{0}.jpg'.format("".join(filename)), 'jpeg')












