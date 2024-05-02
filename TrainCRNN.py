import torch

class CRNNTrainer:
    def __init__(self, model, dataloaders, loss_func, optimizer, epochs, LSTM_step, use_gpu=True):
        self.model = model
        self.training_dataloader, self.testing_dataloader = dataloaders
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epochs = epochs
        self.LSTM_step = LSTM_step
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.training_losses = []
        self.testing_losses = []
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss_accum = self.run_epoch(self.training_dataloader, training=True)
            avg_train_loss = train_loss_accum / len(self.training_dataloader.dataset)
            self.training_losses.append(avg_train_loss)

            self.model.eval()
            with torch.no_grad():
                test_loss_accum = self.run_epoch(self.testing_dataloader, training=False)
            avg_test_loss = test_loss_accum / len(self.testing_dataloader.dataset)
            self.testing_losses.append(avg_test_loss)

            print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}, Testing Loss = {avg_test_loss:.4f}")

    def run_epoch(self, dataloader, training):
        loss_accum = 0.0
        for CNN_in, LSTM_in, HybridNN_out in dataloader:
            CNN_in, LSTM_in, HybridNN_out = CNN_in.to(self.device), LSTM_in.to(self.device), HybridNN_out.to(self.device)
            if training:
                self.optimizer.zero_grad()
            prediction = self.model(LSTM_in, CNN_in, self.LSTM_step)
            loss = self.loss_func(prediction, HybridNN_out)
            if training:
                loss.backward()
                self.optimizer.step()
            loss_accum += loss.item()
        return loss_accum

    def get_losses(self):
        return self.training_losses, self.testing_losses
    
    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        groundtruth = []
        with torch.no_grad():
            for CNN_in, LSTM_in, HybridNN_out in dataloader:
                CNN_in, LSTM_in = CNN_in.to(self.device), LSTM_in.to(self.device)
                prediction = self.model(LSTM_in, CNN_in, self.LSTM_step)
                predictions.append(prediction.cpu())
                groundtruth.append(HybridNN_out)
        return predictions, groundtruth
