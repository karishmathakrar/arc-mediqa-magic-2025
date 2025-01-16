## Setup Instructions

1. **Download data from this google folder:**
https://drive.google.com/file/d/1VOIgXVJy1c8lWFdY63lCyyztY1PS-Bh1/view?usp=sharing

Create an images folder and place the train, test, valid folders in it. 

Structure: 

2024_dataset/
│
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   ├── test/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   └── valid/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...

2. **Make sure you have Python 3.10 installed**  

3. **Create a .venv**

4. **Install packages in requirements.txt**
```bash
pip install -r requirements.txt
```

5. **Create Gemini API key and add it to .env**