import decord
from PIL import Image
def get_image(video_path):
    vr = decord.VideoReader(video_path, width=512, height=512)
    sample_index = list(range(0, len(vr), 1))[:1]
    image_array = vr.get_batch(sample_index)
    image = image_array.asnumpy()[0]
    image = Image.fromarray(image) 
    image.save("./output.png") 



if __name__ == '__main__':
    get_image("/mnt/bn/qy-dcar-valume/Tune-A-Video/wonder woman, wearing a cowboy hat, is skiing.mp4")