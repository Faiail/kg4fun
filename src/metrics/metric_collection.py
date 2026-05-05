from torchmetrics import MetricCollection as Collector


class MetricCollection(Collector):
    def compute(self):
        result = dict()
        for k, metric in self.items():
            result[k] = metric.compute()
        return result