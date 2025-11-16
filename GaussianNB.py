import numpy as np
import pandas as pd
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.gridspec as gridspec
print("All libraries imported. Starting (Training)...")

train_path = "/kaggle/input/fashionmnist/fashion-mnist_train.csv"
test_path = "/kaggle/input/fashionmnist/fashion-mnist_test.csv"

print("\nLoading dataset from CSV...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Splitting data...")
train_labels = train_df['label']
train_images_flat = train_df.drop('label', axis=1)

test_labels = test_df['label']
test_images_flat = test_df.drop('label', axis=1)

print(f"Training on {train_images_flat.shape[0]} samples.")

pipeline = Pipeline([
    ('cleaner', VarianceThreshold(threshold=0.0)),
    ('scaler', StandardScaler()),
    ('pca', PCA(whiten=True)), 
    ('model', GaussianNB())
])

param_grid = {
    'pca__n_components': [80, 100, 120, 140],
    'model__var_smoothing': [1e-9, 1e-8, 1e-7] 
}
total_models = len(param_grid['pca__n_components']) * len(param_grid['model__var_smoothing'])
print(f"Will train {total_models} models on 60,000 samples.")

print("\n--- Starting Model Search (This will be VERY LONG) ---")
start_time = time.time()

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(train_images_flat, train_labels)

end_time = time.time()
print(f"\nSearch completed in {end_time - start_time:.2f} seconds.")

print("\n--- Best Model Found ---")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
               "Coat", "Sandal", "Shirt", "Sneaker", 
               "Bag", "Ankle boot"]

print("\nTraining  is complete")


print("Starting Block 2: Visualization (using your KNN template)...")

try:
    y_pred = best_model.predict(test_images_flat)
    
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='macro', zero_division=0)
    recall = recall_score(test_labels, y_pred, average='macro', zero_division=0)
    f1 = f1_score(test_labels, y_pred, average='macro', zero_division=0)
    
    report = classification_report(test_labels, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(test_labels, y_pred)
    
    print(f"Naive Bayes Model Accuracy: {accuracy:.4f}")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
               'Score': [accuracy*100, precision*100, recall*100, f1*100]}
    bars = ax1.bar(metrics['Metric'], metrics['Score'],
                   color=['#4f46e5','#8b5cf6','#ec4899','#10b981'], edgecolor='black')
    ax1.set_ylim([0,105])
    ax1.set_title('Overall Performance Metrics (Naive Bayes)')
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2., h+1, f'{h:.1f}%', ha='center', fontsize=11)
    
    ax2 = fig.add_subplot(gs[0,1])
    x = np.arange(len(class_names))
    width = 0.25
    ax2.bar(x - width, [report[c]['precision']*100 for c in class_names], width, label='Precision', color='#8b5cf6')
    ax2.bar(x, [report[c]['recall']*100 for c in class_names], width, label='Recall', color='#ec4899')
    ax2.bar(x + width, [report[c]['f1-score']*100 for c in class_names], width, label='F1-Score', color='#10b981')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0,105])
    ax2.set_title('Per-Class Performance (Naive Bayes)')
    
    ax3 = fig.add_subplot(gs[1,:])
    im = ax3.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax3.text(j, i, cm[i,j], ha='center', va='center',
                     color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=9)
    ax3.set_xticks(np.arange(len(class_names)))
    ax3.set_yticks(np.arange(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.set_yticklabels(class_names)
    ax3.set_title('Confusion Matrix (Naive Bayes)')
    plt.colorbar(im, ax=ax3)
    
    ax4 = fig.add_subplot(gs[2,:])
    class_acc = [(y_pred[test_labels==i]==test_labels[test_labels==i]).mean()*100 for i in range(10)]
    bars = ax4.barh(class_names, class_acc, color='#4f46e5', edgecolor='black')
    for i, acc in enumerate(class_acc):
        ax4.text(acc+1, i, f'{acc:.1f}%', va='center', fontsize=10)
    ax4.set_xlim([0,105])
    ax4.set_title('Accuracy by Class (Naive Bayes)')
    
    plt.tight_layout()
    plt.savefig('naive_bayes_fashion_mnist_results.png', dpi=300, bbox_inches='tight')
    plt.show()

except NameError as e:
    print(f"\n!!! ERROR: {e}")
    print("Please make sure you have run the Naive Bayes Training code (Block 1) successfully first.")
