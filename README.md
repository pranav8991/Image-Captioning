
# 🖼️ Project: Image Captioning with Neural Networks 🧠💬

## Welcome to the *Epic Saga* of Image Captioning! 🎉🤖

In this legendary project, I bravely embarked on a quest to build a Neural Network that can look at an image and… wait for it… **TALK ABOUT IT**! Yes, I’ve turned my computer into an art critic, capable of spitting out captions faster than a chatbot on caffeine. ☕💬

Armed with the Microsoft Common Objects in Context [(MS COCO) dataset](https://cocodataset.org/#home), some code, and an unhealthy dose of optimism, I set out to train this model to *understand* images. Spoiler alert: it mostly ended up guessing, but hey, it’s the thought that counts! 😅

![image-captioning](https://github.com/user-attachments/assets/254a36a4-f830-406d-8841-20782b8bd207)
**Image Captioning Model
---

## Introduction 🤓

**Project Overview**:
In this project, you will create a neural network architecture that automatically generates captions for images. After training your network on the[(MS COCO) dataset](https://cocodataset.org/#home), you will test it on *novel images*. Yes, that’s right, I built a machine that invents captions on the fly. 🚀

> “A cat sitting on a skateboard wearing sunglasses” - **My network, probably**

---

## Project Instructions 📝

This epic journey is structured into a series of Jupyter notebooks, each one a stepping stone towards the final goal:

- **0_Dataset.ipynb**: Where I meet the mysterious [(MS COCO) dataset](https://cocodataset.org/#home) and try not to break my internet downloading it. 😬
- **1_Preliminaries.ipynb**: Setting up the tools for battle – neural networks, data loaders, and a lot of coffee. ☕
- **2_Training.ipynb**: The long, hard slog of training, where the GPU becomes my best friend and worst enemy. 🥵
- **3_Inference.ipynb**: Testing the network’s newfound abilities and praying that it doesn’t call a dog a cat. 🙏

All of this is happening in the Udacity workspace, where I can enable GPU mode (a must!) and hopefully not set anything on fire. 🔥

---

## Solver Functionality 💡

| Criteria                              | Submission Requirements                                                                                                                                                    |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🖼️ LSTM Decoder                       | **(AUTOGRADED)** The student correctly implements the LSTM Decoder to handle sequences of image features and words, because why use a simple model when you can use LSTMs? 🤯 |
| 🧠 Image Features Extraction           | **(AUTOGRADED)** The network successfully extracts features from images using a CNN encoder, converting images into something a computer can understand – kind of. 📷➡️🧠   |

---

## How I Survived the Project 🏋️‍♂️

### Step 1: The Beginning of the End 📅

Like all good stories, this one began with downloading a **massive dataset**. I thought, “How hard can it be?” Turns out, very. 😬 After an hour of waiting and some *gentle encouragement* to my internet connection, I was ready to begin.

```bash
(aind) $ git clone https://github.com/udacity/CVND---Image-Captioning-Project.git
```

---

### Step 2: Setting Up the Environment 🛠️

Next, I had to set up the environment, enabling GPU support, and generally trying to make my computer not explode. I created a conda environment, installed dependencies, and held my breath. 💻

```bash
(aind) $ conda create -n image_captioning python=3.6
(aind) $ conda activate image_captioning
(aind) $ pip install -r requirements.txt
```

---

### Step 3: Implementing the CNN Encoder 🖼️➡️🧠

The first task was to create a **CNN Encoder** that could look at an image and turn it into a bunch of numbers. Simple, right? Well, after spending some quality time with PyTorch, I had a basic encoder ready.

```python
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # Using a pre-trained ResNet model to extract features
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
    
    def forward(self, images):
        features = self.resnet(images)
        return features
```

---

### Step 4: Enter the LSTM Decoder 🧠💬

Now came the *fun* part – implementing the **LSTM Decoder**. It’s like giving my computer the ability to speak, but instead of using actual words, it speaks in *vectors*. No, I didn’t completely understand it at first either. 😅

```python
class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNNDecoder, self).__init__()
        # The mystical LSTM, capable of understanding sequences.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs
```

---

### Step 5: The Long, Arduous Training 🏃‍♂️💻

With the architecture ready, it was time to train the model. It only took a **mere** 5-12 hours on the GPU. I made myself a cup of tea, watched the loss go down, and prayed for good results. 🕯️

```python
for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(data_loader):
        outputs = decoder(encoder(images), captions)
        loss = criterion(outputs, captions)
        loss.backward()
        optimizer.step()
```

**Me after 12 hours**: “Is it done yet?”

**GPU**: “Nope.”

**Me**: 😭

---

### Step 6: The Grand Inference 🕵️‍♂️🔍

Finally, it was time to see if my network could actually caption images. I gave it a few test images, crossed my fingers, and ran the code. Did it understand the pictures? Well, sort of. I mean, it’s a work in progress, okay? 😅

```python
image = Image.open('test_image.jpg')
caption = decoder.sample(encoder(image))
print("Generated Caption:", caption)
```

**Network**: “A dog sitting on a bench with a banana.”

**Me**: “Close enough!” 🐶🍌🪑

---

### Step 7: Project Submission Instructions 📤

After countless hours of blood, sweat, and tears (mostly tears), it was time to submit the project. A few quick tips for anyone who dares follow in my footsteps:

1. **Delete Large Files**: Make sure your notebooks are clean, or the submission might be too large to handle.
2. **Check the Rubric**: Review the project rubric to make sure you didn’t miss anything important, like actually training the model.
3. **Click Submit**: And pray. 🙏

---

## How to Run This Masterpiece 💻🎨

1. **Clone the Repo**: Grab this project from GitHub like a boss.
   ```bash
   git clone <repo-url>
   ```

2. **Activate the Environment**: Make sure you're running on the right conda environment.
   ```bash
   conda activate image_captioning
   ```

3. **Navigate to the Project Directory**: You know the drill.
   ```bash
   cd image_captioning_project
   ```

4. **Train the Model**: Make sure you’ve enabled GPU support, and then run the training notebook.
   ```bash
   jupyter notebook 2_Training.ipynb
   ```

5. **Generate Captions**: Use the inference notebook to test your model.
   ```bash
   jupyter notebook 3_Inference.ipynb
   ```

---

## Conclusion 🏁

And that’s how I turned my computer into an amateur poet that describes images in broken English. 🎤✨ It’s been a wild ride full of challenges, but I’m happy to say I came out the other side with a neural network that can almost describe images. Almost.

Remember, no machine is perfect, especially one I built. But hey, we’re all learning, right? 😅 So give this project a try, have fun, and maybe you’ll teach your network to recognize a cat from a dog better than I did.

##Just remember — even the mightiest programmers start with baby steps (and occasional tantrums). And hey, sometimes those baby steps mean seeking help from fellow coders, dissecting their code, and piecing together solutions. So yes, I’ve had my fair share of peeking into others' projects, learning from their work, and figuring out how things tick. It’s all part of the journey. So, maff kar do muja if I borrowed an idea or two along the way—because, in the end, it’s about growing and improving. 😅

---

## License ⚖️

This project is released under the “This Was Harder Than Expected” License. Feel free to fork, modify, and use this code, but remember to credit your fellow strugglers out there! 😆

---
