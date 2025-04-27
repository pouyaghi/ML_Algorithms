from Prepare_data import get_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

x_encoded, y_encoded, encoder = get_data()

# list into tensor
X_tensor = torch.tensor(x_encoded, dtype=torch.long)
Y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# create dataset and dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

# Vocabulary size
vocab_size = len(encoder.classes_) # total number of unique words

class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size= 64, nhead=2, hidden_dim=128, num_layers=2):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x) # (batch_size, seq_length, embed_size)
        embedded = embedded.permute(1,0,2) # Transformer expects (seq_length, batch_size, embed_size)
        transformer_out = self.transformer_encoder(embedded)
        output = transformer_out[-1] # only use the last token's output
        output = self.fc(output)
        return output
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NextWordPredictor(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(dataLoader, leave=True)


    for batch_X, batch_y in loop:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataLoader)
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")


def predict_next_word(model, input_words):
    model.eval()
    input_encoded = encoder.transform(input_words)
    input_tensor = torch.tensor([input_encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predict_index = torch.argmax(output, dim=1).item()
        predicted_word = encoder.inverse_transform([predict_index])[0]

    return predicted_word


# example usage
print(predict_next_word(model, ["the", "quick", "thrown"]))
torch.save(model.state_dict(), "model.pth")