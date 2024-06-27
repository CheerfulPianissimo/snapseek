import pickle
from sklearn.neighbors import NearestNeighbors
from index import model_picker, extract_features,extract_text
import subprocess
filenames = pickle.load(open('./index/filenames.pickle', 'rb'))
feature_list = pickle.load(open('./index/features.pickle',
                                'rb'))
text=pickle.load(open('./index/text.pickle','rb'))

num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)

neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)

model = model_picker()
def find_closest(input_img):
    # input_img=input("Enter file path:")[1:-1]
    features_to_search=extract_features(input_img,model)
    text_in=extract_text(input_img)
    text_no=0
    file_list=[]
    print(text_in)
    if text_in.strip()!="" and text_in.strip()!="desktopwallpapers.net" and len(text_in)>3:
        for i in range(len(text)):
            if text_in.lower().strip() in text[i].lower():
                file_list.append(filenames[i])
                # subprocess.Popen(["rifle", filenames[i]])
                text_no+=1
    distances, indices = neighbors.kneighbors([features_to_search])
    print(text_no,indices[0])
    for i in indices[0][0:4-text_no]:
        print(filenames[i])
        # subprocess.Popen(["rifle", filenames[i]])
        file_list.append(filenames[i])
    return file_list

# from datetime import datetime
# a=datetime.now()
# find_closest('./images/sunset05_1024x768.jpg')
# print(datetime.now()-a)
