import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Đọc dataset
def load_fashion_mnist():
    train_df = pd.read_csv('/kaggle/input/fashion-mnist/fashion-mnist_train.csv')
    test_df  = pd.read_csv('/kaggle/input/fashion-mnist/fashion-mnist_test.csv')
    X_train = train_df.iloc[:, 1:].values 
    y_train = train_df.iloc[:, 0].values
    X_test  = test_df.iloc[:, 1:].values 
    y_test  = test_df.iloc[:, 0].values
    return X_train, y_train, X_test, y_test

# Tải data
X_train_full, y_train_full, X_test_full, y_test_full = load_fashion_mnist()

# Train 60k ảnh, test 10k
n_train, n_test = 60000, 10000
train_idx = np.random.choice(len(X_train_full), n_train, replace=False)
test_idx = np.random.choice(len(X_test_full), n_test, replace=False)
X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
X_test, y_test = X_test_full[test_idx], y_test_full[test_idx]

# Khởi tạo và train mô hình KNN
k = 10
knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) 
knn.fit(X_train, y_train)

# Dự đoán nhãn trên tập test
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Kết quả
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Vẽ biểu đồ tổng hợp
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Biểu đồ cột tổng hợp Accuracy, Precision, Recall, F1
ax1 = fig.add_subplot(gs[0, 0])
metrics = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
           'Score': [accuracy*100, precision*100, recall*100, f1*100]}
bars = ax1.bar(metrics['Metric'], metrics['Score'],
               color=['#4f46e5','#8b5cf6','#ec4899','#10b981'], edgecolor='black')
ax1.set_ylim([0,105])
ax1.set_title('Overall Performance Metrics')
for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x()+bar.get_width()/2., h+1, f'{h:.1f}%', ha='center', fontsize=11)

# Biểu đồ Precision, Recall, F1 theo class
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
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
ax2.set_title('Per-Class Performance')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
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
ax3.set_title('Confusion Matrix')
plt.colorbar(im, ax=ax3)

# Accuracy theo class
ax4 = fig.add_subplot(gs[2,:])
class_acc = [(y_pred[y_test==i]==y_test[y_test==i]).mean()*100 for i in range(10)]
bars = ax4.barh(class_names, class_acc, color='#4f46e5', edgecolor='black')
for i, acc in enumerate(class_acc):
    ax4.text(acc+1, i, f'{acc:.1f}%', va='center', fontsize=10)
ax4.set_xlim([0,105])
ax4.set_title('Accuracy by Class')

# Hiển thị và lưu biểu đồ
plt.tight_layout()
plt.savefig('knn_fashion_mnist_results.png', dpi=300, bbox_inches='tight')
plt.show()
