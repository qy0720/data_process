import os
from PIL import Image

width = 3776
height = 2832
stride = [int(width // 4), int(height // 4)]
num_slices_w = (width - width // 2) // stride[0]  + 1
num_slices_h = (height - height // 2) // stride[1] + 1


predictions = []

image = Image.open('/mnt/bn/qy-dcar-valume/diffusers/ImageInpaint/a.jpg')

for i in range(num_slices_h):  
    for j in range(num_slices_w):  
        # 切割图像块  
        x = j * stride[0]  
        y = i * stride[1] 
 
        slice_img = image.crop((x, y, x + width // 2, y + height // 2))  
             
        predictions.append(slice_img) 


final_prediction = Image.new('RGB', (width, height))  
for i, prediction in enumerate(predictions):  
    x = i % num_slices_w * stride[0]  
    y = i // num_slices_h * stride[1]  

    final_prediction.paste(prediction, (x, y))  

final_prediction.save(os.path.join('/mnt/bn/qy-dcar-valume/diffusers/ImageInpaint/output', 'output.jpg'))