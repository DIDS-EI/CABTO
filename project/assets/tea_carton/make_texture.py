"""Deterministic original green-tea carton art; no external brands or downloads."""
from pathlib import Path
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT=Path(__file__).resolve().parent
W,H=1100,800
rng=random.Random(22)
y,x=np.mgrid[0:H,0:W]
f=.12*np.sin(x/300)+.08*np.cos(y/170)
noise=np.random.default_rng(9).normal(0,1.4,(H,W))
a=np.stack([89+f*130+noise,126+f*175+noise,40+f*95+noise],axis=-1).clip(0,255).astype("uint8")
im=Image.fromarray(a)
leaves=Image.new("RGBA",(W,H));d=ImageDraw.Draw(leaves)
for i in range(80):
    cx=rng.uniform(-100,W+100);cy=rng.uniform(-50,H+100);angle=rng.uniform(-math.pi,math.pi)
    length=rng.uniform(65,230);width=length*rng.uniform(.20,.40)
    axis=np.array([math.cos(angle),math.sin(angle)]);cross=np.array([-axis[1],axis[0]])
    pts=[]
    for j in range(51):
        t=j/50;pt=np.array([cx,cy])+axis*(t-.5)*length+cross*math.sin(math.pi*t)*width*.5;pts.append(tuple(pt))
    for j in range(50,-1,-1):
        t=j/50;pt=np.array([cx,cy])+axis*(t-.5)*length-cross*math.sin(math.pi*t)*width*.5;pts.append(tuple(pt))
    base=rng.choice([(36,77,24,190),(125,164,44,205),(171,193,79,205),(71,118,25,205)])
    d.polygon(pts,fill=base)
    start=tuple(np.array([cx,cy])-axis*length*.48);end=tuple(np.array([cx,cy])+axis*length*.48)
    d.line([start,end],fill=(205,219,124,130),width=2)
    for t in (.22,.36,.50,.64,.78):
        p=np.array([cx,cy])+axis*(t-.5)*length
        for side in (-1,1):
            tip=p+axis*.10*length+cross*side*math.sin(math.pi*t)*width*.38
            d.line([tuple(p),tuple(tip)],fill=(207,218,136,75),width=1)
im=Image.alpha_composite(im.convert("RGBA"),leaves)
d=ImageDraw.Draw(im)
cream=(241,233,163);green=(34,69,26);red=(141,35,36)
d.rectangle((18,18,W-18,H-18),outline=cream,width=3)
d.rectangle((29,29,W-29,H-29),outline=(190,205,101),width=1)
font="/System/Library/Fonts/STHeiti Medium.ttc"
def text(s,xy,size,fill):d.text(xy,s,font=ImageFont.truetype(font,size),fill=fill,anchor="mm")
# Broad cream seal with a dark-green outer ring.
d.ellipse((347,167,753,573),fill=green,outline=cream,width=5)
d.ellipse((366,186,734,554),fill=cream)
d.rounded_rectangle((398,274,702,369),radius=36,fill=red)
text("清  茗",(550,320),57,(255,249,211))
text("绿 茶",(550,438),64,green)
text("GREEN TEA",(550,503),25,green)
text("原叶清香  ·  清新自然",(550,634),31,cream)
d.line((270,678,830,678),fill=cream,width=2)
text("LEAF COLLECTION     /     NET WT. 100 g",(550,708),21,cream)
# Lid seam and edition mark.
text("TEA / 01",(1010,70),21,cream)
im.convert("RGB").save(OUT/"green_tea_front.png")
print(OUT/"green_tea_front.png")
