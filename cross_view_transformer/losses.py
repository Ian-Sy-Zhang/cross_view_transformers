import torch
import logging

from fvcore.nn import sigmoid_focal_loss


logger = logging.getLogger(__name__)


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)


class BinarySegmentationLoss(SigmoidFocalLoss):
    def __init__(
        self,
        label_indices=None,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class CenterLoss(SigmoidFocalLoss):
    def __init__(
        self,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        pred = pred['center']
        label = batch['center']
        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()


class MultipleLoss(torch.nn.ModuleDict):
    """
    losses = MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})
    loss, unweighted_outputs = losses(pred, label)
    """
    def __init__(self, modules_or_weights):
        '''
        在构造函数中，首先分别创建两个新的字典 modules 和 weights，然后遍历输入的字典。
        如果字典的值是浮点数，那么将其作为权重加入 weights 字典；
            否则，将其作为损失函数加入 modules 字典。
        如果没有为某个损失函数指定权重，那么将其权重默认设置为1.0。
        --------
        In the constructor, first create two new dictionaries modules and weights respectively, and then iterate through the input dictionary.
         If the value of the dictionary is a floating point number, then add it to the weights dictionary as a weight;
             Otherwise, add it to the modules dictionary as a loss function.
         If no weight is specified for a loss function, its weight defaults to 1.0.
        '''

        modules = dict()
        weights = dict()

        # Parse only the weights
        for key, v in modules_or_weights.items():
            if isinstance(v, float):
                weights[key.replace('_weight', '')] = v

        # Parse the loss functions
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):
                modules[key] = v

                # Assign weight to 1.0 if not explicitly set.
                if key not in weights:
                    logger.warn(f'Weight for {key} was not specified.')
                    weights[key] = 1.0

        assert modules.keys() == weights.keys()

        super().__init__(modules)

        self._weights = weights

    '''
    forward 方法接受两个参数，pred 和 batch，分别表示预测值和批量数据。
    在该方法中，首先遍历所有的损失函数，使用它们计算损失，并将结果保存在 outputs 字典中。
    然后，根据每个损失的权重计算总损失，并返回总损失和 outputs 字典。
    --------
    The forward method accepts two parameters, pred and batch, which represent predicted values and batch data respectively.
     In this method, first iterate over all loss functions, use them to calculate the loss, and save the results in the outputs dictionary.
     Then, the total loss is calculated based on the weight of each loss, and the total loss and the outputs dictionary are returned.
    '''
    def forward(self, pred, batch):
        '''
        self.items() 遍历 MultipleLoss 类的实例中存储的所有损失函数及其名称。
        对于每一个损失函数 v 和对应的名称 k，调用 v(pred, batch) 计算损失，然后将结果存储在 outputs 字典中。
        在每一个具体的loss function中，会取出batch的label部分（batch分为training data & label）并与prediction计算loss
        --------
        self.items() iterates over all loss functions and their names stored in an instance of the MultipleLoss class.
         For each loss function v and corresponding name k, call v(pred, batch) to calculate the loss and store the result in the outputs dictionary.
         In each specific loss function, the label part of the batch is taken out (batch is divided into training data & label) and the loss is calculated with prediction
        '''
        outputs = {k: v(pred, batch) for k, v in self.items()}
        total = sum(self._weights[k] * o for k, o in outputs.items())

        return total, outputs
