import os
import numpy as np
from PIL import Image
from skimage import color, filters, measure
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

tumor_types = {
    'adenocarcinoma': 'Non-Small Cell Lung Cancer (NSCLC) - Adenocarcinoma',
    'squamous_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Squamous Cell Carcinoma',
    'large_cell': 'Non-Small Cell Lung Cancer (NSCLC) - Large Cell Carcinoma',
    'small_cell': 'Small Cell Lung Cancer (SCLC)',
    'metastatic': 'Secondary Lung Tumors - Metastatic',
    'hamartoma': 'Benign Lung Tumors - Hamartomas',
    'pulmonary_adenoma': 'Benign Lung Tumors - Pulmonary Adenomas'
}

def print_tumor_types():
    print("Tumor Types:")
    for key, value in tumor_types.items():
        print(f"{key}: {value}")

image_directory = "./"

def load_images_and_extract_features(image_directory):
    features = []
    labels = []
    
    for image_file in os.listdir(image_directory):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_directory, image_file)
            image = Image.open(image_path)
            image_array = np.array(image)

            if image_array.shape[-1] == 4:
                image_array = color.rgba2rgb(image_array)

            image_gray = color.rgb2gray(image_array)
            image_normalized = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))

            mean_intensity = np.mean(image_normalized)
            std_intensity = np.std(image_normalized)
            cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            edges_sobel = filters.sobel(image_normalized)
            avg_edge_strength = np.mean(edges_sobel)

            glcm = graycomatrix((image_normalized * 255).astype(np.uint8), distances=[1], angles=[0], 
                                levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            asm = graycoprops(glcm, 'ASM')[0, 0]

            label = image_file.split('_')[0]
            features.append([mean_intensity, std_intensity, cv, avg_edge_strength, contrast, 
                             dissimilarity, homogeneity, energy, correlation, asm])
            labels.append(label)

    return np.array(features), np.array(labels)

X, y = load_images_and_extract_features(image_directory)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

def visualize_predictions(X_test, y_test, y_pred):
    plt.figure(figsize=(15, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i].reshape(512, 512), cmap='gray')
        plt.title(f'True: {y_test[i]}\nPred: {y_pred[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(X_test, y_test, y_pred)

def tumor_likelihood(image_features):
    mean_intensity, std_intensity, cv, avg_edge_strength, contrast, dissimilarity, homogeneity, energy, correlation, asm = image_features
    weights = {
        'mean_intensity': 0.2,
        'std_intensity': 0.2,
        'cv': 0.1,
        'avg_edge_strength': 0.2,
        'contrast': 0.1,
        'dissimilarity': 0.1,
        'homogeneity': 0.1,
        'energy': 0.1,
    }
    T = (weights['mean_intensity'] * mean_intensity +
         weights['std_intensity'] * std_intensity +
         weights['cv'] * cv +
         weights['avg_edge_strength'] * avg_edge_strength +
         weights['contrast'] * contrast +
         weights['dissimilarity'] * dissimilarity +
         weights['homogeneity'] * homogeneity +
         weights['energy'] * energy)
    return T

sample_image_features = X_test[0]
tumor_score = tumor_likelihood(sample_image_features)
print(f"Tumor Likelihood Score: {tumor_score:.4f}")

threshold = 0.5
classification = "Tumor" if tumor_score >= threshold else "Not Tumor"
print(f"Classification based on likelihood score: {classification}")

def determine_tumor_type(prediction):
    if classification == "Tumor":
        if prediction in tumor_types:
            print(f"Detected Tumor Type: {tumor_types[prediction]}")
        else:
            print("Tumor type not recognized.")
    else:
        print("No tumor detected.")

determine_tumor_type(y_pred[0])

def highlight_tumor(image, edges):
    labeled_image = measure.label(edges > 0.1)
    highlighted_image = color.label2rgb(labeled_image, image=image, bg_label=0)
    return highlighted_image

def plot_highlighted_tumor(X_test, index):
    original_image = X_test[index].reshape(512, 512)
    edges = filters.sobel(original_image)
    highlighted_image = highlight_tumor(original_image, edges)
    plt.figure(figsize=(8, 8))
    plt.imshow(highlighted_image)
    plt.title("Highlighted Tumor Regions")
    plt.axis('off')
    plt.show()

plot_highlighted_tumor(X_test, 0)

def plot_histogram(image):
    plt.figure(figsize=(10, 5))
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Intensity Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

plot_histogram(X_test[0].reshape(512, 512))

def plot_edge_detection(image_normalized, edges):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_normalized, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='hot')
    plt.title('Sobel Edge Detection Result')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_edge_detection(X_test[0].reshape(512, 512), filters.sobel(X_test[0].reshape(512, 512)))

def plot_glcm_features(glcm):
    features = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    values = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=features, y=values)
    plt.title('GLCM Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.grid()
    plt.show()

glcm = graycomatrix((X_test[0].reshape(512, 512) * 255).astype(np.uint8), distances=[1], angles=[0], 
                    levels=256, symmetric=True, normed=True)
plot_glcm_features(glcm)

print_tumor_types()