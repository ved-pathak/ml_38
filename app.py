from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageDraw, ImageFont,ImageFilter
import requests
import os
from io import BytesIO
import random
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline





app = Flask(__name__)

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3-medium-diffusers"
headers = {"Authorization": "Bearer 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"}

def query_img(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print("Error querying Hugging Face API:", response.status_code, response.text)
        return None
    return response.content

# For text extractor
API_URL_extractor = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

def query_ext(payload):
    response = requests.post(API_URL_extractor, headers=headers, json=payload)
    return response.json()


# @app.route('/')
# def land():
#     return render_template('landing.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    ###phi model loading
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",

        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    def msg_gen(prompt):
        msg_gen="generate a greeting message for a card in 10 words related to:"+prompt
        
        messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "generate the greeting message in 10 words for given event : 'a birthday card for ved'"},
        {"role": "assistant", "content": "Happy Birthday Ved! Wishing you joy, health, and happiness!"},
        {"role": "user", "content": msg_gen},
        ]
        output1 = pipe(messages, **generation_args)
        print(output1[0]['generated_text'])
        message =output1[0]['generated_text']
        return message
    def happy_msggen(prompt):
        happy_gen_in="wish for the ocassion in 2-3 words related to:"+prompt
        happy_gen = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "wish for the ocassion in 2 words related to:'a birthday card for ved'"},
        {"role": "assistant", "content": "Happy Birthday!"},
        {"role": "user", "content": happy_gen_in},
        ]
        output = pipe(happy_gen, **generation_args)
        print(output[0]['generated_text'])
        text=output[0]['generated_text']
        return text
    
    def img_prompt_gen(prompt):
        image_gen_in="enhance the prompt to a very detailed prompt of about 100 words for text to image generation model to generate a 'background design' related to:"+prompt

        image_gen_prompt= [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "enhance the prompt to a very detailed prompt of about 100 words for text to image generation model to generate a 'background design' related to:'a birthday card for my friend VED with some personalised message on it'"},
        {"role": "assistant", "content": "Create a vibrant and festive background design for a birthday card. Use a bright, cheerful color palette with a base of turquoise or light purple. Decorate the background with playful elements like colorful balloons, confetti, and streamers scattered throughout. Incorporate festive icons such as a multi-layered birthday cake, wrapped gifts, and party hats. Add details like stars, polka dots, and sparkles to enhance the celebratory atmosphere. The design should be dynamic and joyful, exuding warmth and excitement, suitable for a lively birthday celebration. The overall feel should be fun, festive, and inviting, ready to complement a personalized birthday message."},
        {"role": "user", "content": "enhance the prompt to a very detailed prompt of about 100 words for text to image generation model to generate a 'background design' related to:'a anniversary card for nobita and shizuka and add some greetings'"},
        {"role": "assistant", "content": "Design a romantic and elegant background for an anniversary card dedicated to Nobita and Shizuka. Use a soft, pastel color palette with shades of blush pink, lavender, or ivory. The background should feature delicate elements such as hearts, roses, and subtle lace patterns. Add a touch of gold or silver accents to enhance the sophisticated look. Incorporate symbols of love and commitment, like intertwined rings or a silhouette of a couple. Scatter petals and sparkling lights to create a dreamy, magical atmosphere. Include a space at the top or bottom for a greeting message, styled in an elegant script font, such as 'Happy Anniversary, Nobita and Shizuka! Wishing you a lifetime of love and happiness.' The design should evoke a sense of warmth, romance, and celebration, perfect for commemorating a special anniversary."},

        {"role": "user", "content":image_gen_in},
        ]

        output_fin1 = pipe(image_gen_prompt, **generation_args)

        print(output_fin1[0]['generated_text'])
        prompt_fin_image=output_fin1[0]['generated_text']
        return prompt_fin_image

     # Query the Hugging Face model API to generate image
    def gen_img(prompts):
        image_bytes = query_img({"inputs": prompts})
        if image_bytes is None:
            return "Error generating image"
        # Query the Hugging Face model API to generate image

        image = Image.open(BytesIO(image_bytes))
        return image    
    #generating msg1
    message=msg_gen(prompt) 
    message2=msg_gen(prompt)
    text = happy_msggen(prompt)
    prompt_fin_image = img_prompt_gen(prompt)
    prompt_fin_image2 = img_prompt_gen(prompt)
    image=gen_img(prompt_fin_image)   
    image2=gen_img(prompt_fin_image2)

   

    # Save the base image
    image_path = os.path.join('static', 'generated', 'base_image.png')
    image.save(image_path)
    image_path2 = os.path.join('static', 'generated', 'base_image2.png')
    image2.save(image_path2)

    base_image = Image.open(image_path)
    base_image2 = Image.open(image_path2)
   
    draw = ImageDraw.Draw(base_image)
    draw2 = ImageDraw.Draw(base_image2)

    width = base_image.width

    font_paths = {
        'Comic Sans': os.path.join('fonts', 'Comic Sans MS.ttf'),
        'air': os.path.join('fonts', 'AirtravelerspersonaluseBdit-ow59x.otf'),
        'gold': os.path.join('fonts', 'GoldenbeachpersonaluseBdit-jEgDM.otf'),
        'ice': os.path.join('fonts', 'IcecreamypersonaluseBold-3zgoy.otf'),
        'ignite': os.path.join('fonts', 'IgnitetypepersonaluseRegular-K791Z.otf'),
        'island': os.path.join('fonts', 'IslandsitalicpersonaluseIta-gxgB4.otf'),
        'nature': os.path.join('fonts', 'NatureBeautyPersonalUse-9Y2DK.ttf'),
        'plaza': os.path.join('fonts', 'PlazapersonaluseBold-d91XR.otf'),
        'quick': os.path.join('fonts', 'QuicksilverFastRegular-DO3oE.ttf'),
        'rich': os.path.join('fonts', 'RichTheBarberPersonalUse-gx824.ttf'),
        'sung': os.path.join('fonts', 'SunglasstypepersonaluseBold-Ea7El.otf'),
        'wesay': os.path.join('fonts', 'WesaysopersonaluseBold-L3oOZ.otf'),
    }
    
    font_path = random.choice(list(font_paths.values()))
    font_path2 = random.choice(list(font_paths.values()))
    font2_path = random.choice(list(font_paths.values()))
    font2_path2 = random.choice(list(font_paths.values()))
    font_size2=random.randint(150, 250)
    font2 = ImageFont.truetype(font_path2, font_size2)
    font22 = ImageFont.truetype(font2_path2, font_size2)


    print(font_path)
    font_size = random.randint(150, 200)
    font = ImageFont.truetype(font_path, font_size)
    font21 = ImageFont.truetype(font2_path, font_size)
    text_width1 = font.getlength(text)
    text2_width1=font21.getlength(text)
    max_x = int(width - text_width1)
    max_x2= int(width- text2_width1)

    while max_x < 20:
        font_size = random.randint(100, 200)
        font = ImageFont.truetype(font_path, font_size)
        text_width1 = font.getlength(text)
        max_x = int(width - text_width1)


    #def detect_dominant_color(image_path, region, k=4, image_resize=(150, 150)):
        #image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if region:
        #     x, y, w, h = region
        #     image = image[y:y+h, x:x+w]
        # image = cv2.resize(image, image_resize, interpolation=cv2.INTER_AREA)
        # pixels = image.reshape(-1, 3)
        # kmeans = KMeans(n_clusters=k)
        # kmeans.fit(pixels)
        # counts = np.bincount(kmeans.labels_)
        # dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
        # return dominant_color

    #def get_contrasting_color(dmcolor):
        # dmcolor = np.asarray(dmcolor)
        # contrasting_color = 255 - dmcolor
        # return contrasting_color

    # dom_color = detect_dominant_color(image_path, draw.textbbox((0, 0), text, font=font), k=4, image_resize=(150, 150))
    # contrast_color = get_contrasting_color(dom_color)
    # color = tuple(map(int, contrast_color))
    def get_random_color():
        """
        Generate a random RGB color.
        Returns:
            tuple: A tuple representing the RGB color (R, G, B).
        """
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r, g, b)
    def percent(a, max):
        return (a / max) * 100

   
    def add_text_to_image(image, text, text_color, dim, font1,size):
        # Define font and initial font size
        font_size = size
        font = font1

        # Define initial text position
        text_position = [0.5 * dim[0], 0.6 * dim[1]]  # X, Y coordinate
        draw = ImageDraw.Draw(image)
        lines = []

        while True:
            max_width = dim[0] - 2 * text_position[0]  # Maximum text width (image width - margin)
            max_h = (percent(max_width, dim[0]) * dim[1]) / 100

            text_position[1] = (dim[1] - max_h) / 2
            words = text.split()
            line = []
            h_max = 0

            for word in words:
                line.append(word)
                bbox = draw.textbbox((0, 0), ' '.join(line), font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                if w > max_width:
                    line.pop()
                    lines.append(' '.join(line))
                    line = [word]
                    h_max += h

                if len(lines) == 2:
                    line = words[len(lines) - 1:]  # Remaining words for the second line
                    break

            lines.append(' '.join(line))
            h_max += h

            if h_max > max_h:
                text_position[0] -= (8 * dim[0]) / 100
                continue
            else:
                if len(lines) > 2:
                    font_size -= 10  # Reduce font size if text doesn't fit in two lines
                    font = ImageFont.truetype(font_path, font_size)
                    lines = []
                else:
                    break
                    

        y = text_position[1] + 500  # Adjusted to move text downward
        text_bbox = draw.textbbox((text_position[0], y), '\n'.join(lines), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Create a mask for the text background blur
        text_background = image.crop((text_position[0], y, text_position[0] + text_width, y + text_height))
        blurred_background = text_background.filter(ImageFilter.GaussianBlur(radius=10))
        image.paste(blurred_background, (int(text_position[0]), int(y)))

        for line in lines:
            draw.text((text_position[0], y), line, fill=text_color, font=font)
            bbox = draw.textbbox((text_position[0], y), line, font=font)
            line_height = bbox[3] - bbox[1]
            y += line_height

    color1 = get_random_color()
    color2 = get_random_color()
    color3 = get_random_color()
    color4 = get_random_color()


    width, height = base_image.size
    dim = (width, height)
     # White color in RGB
    
    bbox2 = draw.textbbox((0, 0), text, font=font2)
    text_width1 = bbox2[2] - bbox2[0]
    x=((width-text_width1)/2)+100
    y2=random.randint(100, int(0.4*height))
    position2 = (x,y2)


    add_text_to_image(base_image, message,color1, dim,font,font_size)
    add_text_to_image(base_image2, message2,color3, dim,font21,font_size)
    draw.text(position2, text, font=font2, fill=color2)
    draw.text(position2, text, font=font22, fill=color4)
    


    final_image_path = os.path.join('static', 'generated', 'final_image_with_text.png')
    base_image.save(final_image_path)

    final_image_path2 = os.path.join('static', 'generated', 'final_image_with_text2.png')
    base_image2.save(final_image_path2)


    #using sd3 prompt enhancement
    def enhanced_prompt(prompt):
        enh_prompt = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "enhance a detailed prompt for text to image generation modelto geneate a greeting card 'a anniversary card for ved and harsh  and add some greetings '"},
            {"role": "assistant", "content": "Create an elegant and heartfelt anniversary card for Ved and Harsh. The card should feature a beautifully illustrated background with a subtle gradient of soft pastel colors, symbolizing the blending of their lives. In the center, place a hand-drawn illustration of a couple holding hands, smiling, and surrounded by delicate flowers and a small tree, representing growth and love. The couple's names, 'Ved' and 'Harsh,' should be elegantly written in cursive script along the bottom of the card. Above the couple, include a heartfelt message that reads, 'Happy Anniversary, Ved and Harsh! Wishing you a joyous anniversary filled with love and cherished memories.' The overall design should be sophisticated yet warm, with a touch of whimsy, and the greetings should convey genuine affection and well wishes for their special day."},
            {"role": "user", "content": prompt},
        ]

        output_fin = pipe(enh_prompt, **generation_args)
        return output_fin[0]['generated_text']
    prompt_enh1=enhanced_prompt(prompt)
    image_final3=gen_img(prompt_enh1)#1st img gen
    final_image_path3 = os.path.join('static', 'generated', 'final_image_with_text3.png')
    image_final3.save(final_image_path3)#img save

    prompt_enh2=enhanced_prompt(prompt)
    image_final4=gen_img(prompt_enh2)#2st img gen
    final_image_path4 = os.path.join('static', 'generated', 'final_image_with_text4.png')
    image_final4.save(final_image_path4)#img save


    return render_template('result.html')


if __name__ == '__main__':
    if not os.path.exists('static/generated'):
        os.makedirs('static/generated')
    if not os.path.exists('fonts'):
        os.makedirs('fonts')
    # Make sure to have the font files in the 'fonts' directory
    app.run(debug=True)
