import pickle
from sklearn.neighbors import NearestNeighbors
from index import model_picker, extract_features
filenames = pickle.load(open('./index/filenames.pickle', 'rb'))
feature_list = pickle.load(open('./index/features-resnet.pickle',
                                'rb'))

num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)

neighbors = NearestNeighbors(n_neighbors=5,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)

model_architecture = 'resnet'
model = model_picker(model_architecture)


features_to_search=extract_features(input("Enter file path:")[1:-1],model)
distances, indices = neighbors.kneighbors([features_to_search])

for i in indices[0]:
    print(filenames[i])