from torch.utils.data import Dataset


class HeatmapDataset(Dataset):

    def __init__(self, dataset=None, heatmaps=None):
        super(HeatmapDataset, self).__init__()

        self.dataset = dataset
        self.heatmaps = heatmaps
        assert (len(self.dataset) == len(self.heatmaps)), f"Found {len(self.dataset)} instances but {len(self.heatmaps)} heatmaps"

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'heatmap': self.heatmaps[item]
        }

    def __len__(self):
        return len(self.dataset)