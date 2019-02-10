import numpy as np

class ConfusionMatrix(object):
    def __init__(self):
        self.data = np.zeros(shape=(3, 3))
        self.true_values = np.zeros(shape=(3))
    
    def insert(self, true_idx, pred_idx):
        self.data[true_idx, pred_idx] += 1
        self.true_values[true_idx] += 1
    
    def __str__(self):
        m = self.data / np.sum(self.data)

        s = """
                  pred H  pred D  pred A
           true H   {:.2f}  {:.2f}  {:.2f}
           true D   {:.2f}  {:.2f}  {:.2f}
           true A   {:.2f}  {:.2f}  {:.2f}                   
        """.format(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2],
                   m[2, 0], m[2, 1], m[2, 2])

        return s

    def get_acc(self):
        m = self.data / np.sum(self.data)
        return m[0, 0] + m[1, 1] + m[2, 2] 
    
    def recall(self, label):
        m = self.data

        tp = m[label, label]
        fn = m[label, 0] + m[label, 1] + m[label, 2] - m[label, label]
        
        return float(tp) / float(tp + fn)
    
    def precision(self, label):
        m = self.data 
    
        tp = m[label, label]
        fp = m[0, label] + m[1, label] + m[2, label] - m[label, label]

        return float(tp) / float(tp + fp)

    def class_acc(self, label):
        num = self.data[label, label]
        den = self.true_values[label]

        return float(num) / float(den)






