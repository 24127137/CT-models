import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

def load_fashion_mnist():
    # ĐÚNG đường dẫn Kaggle Input (theo screenshot)
    train_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
    test_df  = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

    # X = pixel, y = label
    X_train = train_df.iloc[:, 1:].values / 255.0
    y_train = train_df.iloc[:, 0].values

    X_test  = test_df.iloc[:,  1:].values / 255.0
    y_test  = test_df.iloc[:,  0].values

    return X_train, y_train, X_test, y_test

# LOAD FULL DATA
X_train, y_train, X_test, y_test = load_fashion_mnist()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# DECISION TREE
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
# Predict
y_pred = dt.predict(X_test)

# METRICS
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1        = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("Decision Tree Result")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")



# LABEL NAMES
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
# PLOTTING RESULTS
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)



# 1. Overall Metrics
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores  = [accuracy*100, precision*100, recall*100, f1*100]

bars = ax1.bar(metrics, scores,
               color=['#4f46e5','#8b5cf6','#ec4899','#10b981'],
               edgecolor='black')

ax1.set_ylim([0, 105])
ax1.set_title('Decision Tree - Overall Performance')

for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%",
             ha='center', fontsize=12)



# 2. Per-Class Metrics
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
ax2 = fig.add_subplot(gs[0, 1])

x = np.arange(len(class_names))
width = 0.25

ax2.bar(x - width, [report[c]['precision']*100 for c in class_names],
        width, label='Precision', color='#8b5cf6')
ax2.bar(x, [report[c]['recall']*100 for c in class_names],
        width, label='Recall', color='#ec4899')
ax2.bar(x + width, [report[c]['f1-score']*100 for c in class_names],
        width, label='F1-Score', color='#10b981')

ax2.set_xticks(x)
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.set_ylim([0, 105])
ax2.legend()
ax2.set_title('Per-Class Performance')

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ax3 = fig.add_subplot(gs[1, :])

im = ax3.imshow(cm, cmap='Blues')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax3.text(j, i, cm[i, j],
                 ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')

ax3.set_xticks(np.arange(len(class_names)))
ax3.set_yticks(np.arange(len(class_names)))
ax3.set_xticklabels(class_names, rotation=45, ha='right')
ax3.set_yticklabels(class_names)
ax3.set_title("Confusion Matrix")

plt.colorbar(im, ax=ax3)



# 4. Accuracy per Class
ax4 = fig.add_subplot(gs[2, :])
class_acc = [(y_pred[y_test == i] == y_test[y_test == i]).mean()*100 for i in range(10)]

bars = ax4.barh(class_names, class_acc, color='#4f46e5', edgecolor="black")

for i, acc in enumerate(class_acc):
    ax4.text(acc + 1, i, f"{acc:.1f}%", va='center')

ax4.set_xlim([0, 105])
ax4.set_title("Accuracy per Class")

# ---------------------------
plt.tight_layout()
plt.savefig("decision_tree_results.png", dpi=300)
plt.show()
