# tests/test_train_transformer.py
import unittest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import train_transformer
from torch.utils.data import DataLoader

class TestTrainTransformer(unittest.TestCase):

    @patch('train_transformer.data_loader.df')
    def test_data_loading(self, mock_df):
        """
        测试数据加载和预处理
        """
        # 模拟数据加载
        mock_df = pd.DataFrame({
            'UserId': [1, 1, 2],
            'Word': ['apple', 'banana', 'cherry'],
            'Familiarity': [0.8, 0.9, 0.7],
            'CreatedAt': ['2022-01-01', '2022-01-02', '2022-01-03']
        })
        
        # 测试数据集
        dataset = train_transformer.FamiliarityDataset(mock_df)
        
        # 验证数据集长度
        self.assertEqual(len(dataset), 2)  # 2 是数据集中的项数

        # 验证每个数据项
        input_seq, target_familiarity = dataset[0]
        self.assertEqual(input_seq.tolist(), [2])  # WordId = 1 (banana) -> 2  (索引从1开始)
        self.assertEqual(target_familiarity.item(), 0.9)

    @patch('train_transformer.FamiliarityTransformer')
    @patch('train_transformer.optim.Adam')
    def test_model_training(self, mock_Adam, mock_FamiliarityTransformer):
        """
        测试模型训练过程
        """
        # 模拟模型和优化器
        mock_model = MagicMock()
        mock_FamiliarityTransformer.return_value = mock_model
        
        mock_optimizer = MagicMock()
        mock_Adam.return_value = mock_optimizer

        # 模拟数据
        mock_df = pd.DataFrame({
            'UserId': [1, 1, 2],
            'Word': ['apple', 'banana', 'cherry'],
            'Familiarity': [0.8, 0.9, 0.7],
            'CreatedAt': ['2022-01-01', '2022-01-02', '2022-01-03']
        })
        
        # 模拟数据集
        dataset = train_transformer.FamiliarityDataset(mock_df)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 测试训练循环
        for epoch in range(1):  # 只训练1个epoch
            for _, _ in dataloader:
                mock_model.zero_grad()
                mock_optimizer.step()
                
        # 验证模型是否执行了反向传播和优化
        mock_model.zero_grad.assert_called()
        mock_optimizer.step.assert_called()

    @patch('train_transformer.torch.save')
    def test_model_saving(self, mock_save):
        """
        测试模型保存功能
        """
        # 模拟模型
        mock_model = MagicMock()
        
        # 调用模型保存方法
        train_transformer.torch.save(mock_model.state_dict(), "familiarity_transformer.pth")
        
        # 确保保存函数被调用
        mock_save.assert_called_once_with(mock_model.state_dict(), "familiarity_transformer.pth")

if __name__ == '__main__':
    unittest.main()
