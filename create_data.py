import os
import string
import numpy as np
from PIL import ImageFont, ImageDraw, Image

#############################
img_width = 128
img_height = 128
font_size = 120
white = (255, 255, 255)
balck = (0, 0, 0)
fonts_files = [ os.path.join(i) for i in os.listdir( os.path.join('fonts') )]
count = 0
characters_pt = "áàãâÀÁÃÂçÇóòõôÔÒÓÕéèẽêÈÉÊẼúùÙÚªº"
characters = string.digits + string.ascii_letters + string.punctuation + characters_pt
#############################

def make_char_font( char, font_name, count, folder ):
  try:
    font = ImageFont.truetype(os.path.join("fonts", font_name), font_size)
    image = Image.new("RGB", (img_width, img_height), (255,255,255))
    draw = ImageDraw.Draw(image)

    text_width, text_height = draw.textsize(char, font)
    position_x =  (img_width - text_width)/2
    position_y = (img_height - text_height)

    if (position_y < 0):
        position_y = position_y * 2
    else:
        position_y = position_y / 2

    draw.text((position_x, position_y), char, (0,0,0), font=font, align='center')
    image.save( os.path.join("data", f"{folder}" ,f"0000{count}.jpg" ) )
  except:
    pass


def make_font(font_name, count):
  for character in characters:
    if (character == '/'):
      folder = 'barra'
    elif (character == '.'):
      folder = 'ponto'
    else:
      folder = character

    try:
      os.makedirs(os.path.join( "data", f"{folder}" ))
    except:
      pass
    make_char_font( character, font_name, count, folder )

for font_name in fonts_files:
  make_font(font_name, count)
  count = count+1
