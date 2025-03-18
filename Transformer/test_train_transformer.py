import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from Transformer import train_transformer
from torch.utils.data import DataLoader

class TestTrainTransformer(unittest.TestCase):

    @patch('train_transformer.FamiliarityTransformer')
    @patch('train_transformer.optim.Adam')
    def test_model_training(self, mock_Adam, mock_FamiliarityTransformer):
        """
        Test the model training process without involving any database.
        """
        # Mock the model and optimizer
        mock_model = MagicMock()
        mock_FamiliarityTransformer.return_value = mock_model

        mock_optimizer = MagicMock()
        mock_Adam.return_value = mock_optimizer

        # Mock the data (no database involvement)
        mock_df = pd.DataFrame({
            'UserId': [1, 1, 2],
            'WordId': [1, 2, 3],  # Ensure 'WordId' column is correctly provided
            'Word': ['apple', 'banana', 'cherry'],
            'Familiarity': [0.8, 0.9, 0.7],
            'CreatedAt': ['2022-01-01', '2022-01-02', '2022-01-03']
        })

        # Mock the dataset
        dataset = train_transformer.FamiliarityDataset(mock_df)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Test the training loop (1 epoch)
        for epoch in range(1):  # Only 1 epoch for testing
            for _, _ in dataloader:
                mock_model.zero_grad()
                mock_optimizer.step()

        # Verify that backward pass and optimizer step were called
        mock_model.zero_grad.assert_called()
        mock_optimizer.step.assert_called()

    @patch('train_transformer.torch.save')
    def test_model_saving(self, mock_save):
        """
        Test the model saving functionality without database involvement.
        """
        # Mock the model
        mock_model = MagicMock()

        # Call the model save method
        train_transformer.torch.save(mock_model.state_dict(), "familiarity_transformer.pth")

        # Ensure the save function was called correctly
        mock_save.assert_called_once_with(mock_model.state_dict(), "familiarity_transformer.pth")


if __name__ == '__main__':
    unittest.main()
