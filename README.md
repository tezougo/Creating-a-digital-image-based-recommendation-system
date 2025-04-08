# Image-Based Product Recommendation System using Deep Learning

This project aims to build an intelligent **image recommendation system** capable of identifying and retrieving visually similar products based on input images. Unlike conventional systems that rely on textual metadata (such as brand, price, or description), this project uses the **visual appearance** of products (shape, color, texture) to make recommendations.

---

## Objective

Develop and fine-tune a **deep learning model using Transfer Learning** on a custom dataset with four product categories — **watches, t-shirts, bicycles, and shoes**. The model learns to classify these categories and then serves as an **image encoder** to generate **embeddings**, which are used to find visually similar products.

---

## 📁 Project Structure

```
project-root/
│
├── data/                     # Downloaded dataset organized in 4 folders by class
│   ├── relogio/              # Watch images
│   ├── camiseta/             # T-shirt images
│   ├── bicicleta/            # Bicycle images
│   └── sapato/               # Shoe images
│
├── embeddings/               # Where image embeddings and visual results are stored
    └── image_embeddings.pkl  # Saved vectors with image paths and class labels
├── model_recommendation.keras # Trained model using VGG16 + custom layers
├── notebooks/                # Where notebook implementation are stored
    └── Creating_a_digital_image_based_recommendation_system.ipynb            # Full implementation in Google Colab
└── README.md
```

---

## Technologies & Libraries Used

| Technology             | Purpose                                                  |
|------------------------|----------------------------------------------------------|
| **TensorFlow / Keras** | Transfer Learning with VGG16, training, fine-tuning, and inference |
| **NumPy / Pandas**     | Embedding processing, data inspection, storage           |
| **Scikit-learn**       | Cosine similarity and vector ranking                     |
| **Matplotlib / PIL**   | Visualization of image predictions and recommendations  |
| **Google Colab**       | Free GPU runtime (Tesla T4) for end-to-end development   |

---

## How It Works

### 1. Dataset Preparation  
- Images are collected from **KaggleHub** and organized into four folders (one per class).  
- Each class contains up to 200 curated images ensuring **visual coherence** and variation.

### 2. Data Augmentation & Preprocessing  
- Images are resized and normalized.  
- We apply real-time data augmentation to prevent overfitting during training.

### 3. Transfer Learning Model  
- A **pre-trained VGG16** model is used as the base.  
- We frozen the down layers and add custom dense layers for **4-class classification**.   
- Initial training is done with frozen convolutional layers.

### 4. Fine-Tuning (optional)
- After initial convergence, we unfreeze the down layers and **retrain with a lower learning rate**.  
- This improves **feature adaptation** to our domain-specific dataset.

### 5. Model Evaluation  
- Training history is analyzed via accuracy/loss plots.  
- Final model is saved for future embedding generation and inference.

### 6. Image Classification Test  
- A single image can be classified using the trained model.  
- Results are shown visually with predicted class labels.

### 7. Feature Extraction (Embeddings)  
- The model’s second-to-last layer is used to extract **high-dimensional image vectors**.  
- Each image is passed through the encoder and its embedding is saved.

### 8. Similarity Search & Recommendation  
- When a query image is given, we extract its embedding.  
- We compare it to all other embeddings using **cosine similarity** and rank the results.

### 9. Visual Recommendation Output  
- The top N visually similar images are displayed alongside the query image, including similarity scores and labels.

---

## Example Output

<p align="center">
  <img src="https://github.com/user-attachments/assets/3d75205c-de3c-4e31-b68f-dda4e412b1ad" width="600" />
</p>

> Top 5 visually similar products returned based on appearance — **not metadata**.

---

## Try It Yourself (Google Colab)

You can run this project from start to finish using the notebook below:

[🔗 Open in Google Colab](https://colab.research.google.com/drive/1RRbDRjbgGB4W9YbsRGKgmWB80ErnwBhS#scrollTo=9wdBcj47fN9V)

---

## Improvements and Suggestions

- Extend dataset with new classes (e.g., bags, glasses, hats)
- Export model to **TensorFlow Lite or ONNX** for deployment on edge devices

---

## Contributions

Contributions are welcome! If you wish to propose improvements:

```bash
git checkout -b new-feature
git commit -m "feat: add new similarity method"
git push origin new-feature
```

Then submit a **Pull Request**

---

## 📜 License

This project is licensed under the **MIT License**.
