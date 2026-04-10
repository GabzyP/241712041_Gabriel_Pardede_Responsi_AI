from flask import Flask, request, render_template
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 16 * 16, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.output = nn.Linear(256, 3)

    def forward(self, x):
        x = self.relu(self.bn1(self.pooling(self.conv1(x))))
        x = self.relu(self.bn2(self.pooling(self.conv2(x))))
        x = self.relu(self.bn3(self.pooling(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.bn_fc(self.linear(x))))
        x = self.output(x)
        return x

app = Flask(__name__)
device = torch.device('cpu')

model = Net()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_names = ['cat', 'dog', 'wild']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream).convert('RGB')
            img_tensor = eval_transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = torch.argmax(output, 1).item()
                hasil = class_names[pred_idx]
            
            return render_template('index.html', hasil_prediksi=hasil)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)