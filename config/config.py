class Config(object):
    """配置类"""

    def __init__(self):
        # path config
        self.pdf_path = "pdf/"
        self.img_path = "img/"
        self.temp_path = 'temp/'
        self.txt_path = 'txt/'
        self.dict_path = "dataset/drawing_dict.txt"
        self.label_file = 'dataset/tag.txt'
        self.train_file = 'dataset/train/'
        self.dev_file = 'dataset/test/'
        self.test_file = 'dataset/test/'
        self.log_dir = 'log/'
        # self.vocab = 'dataset/vocab.txt'
        self.vocab = 'model/RoBERTa_zh_L12_PyTorch/vocab.txt'
        self.pretrain_model_name = 'model/RoBERTa_zh_L12_PyTorch/'
        self.target_dir = 'output/'
        # value config
        self.base_size = 3840
        # model config
        self.max_length = 512
        self.batch_size = 4
        self.shuffle = True
        self.rnn_hidden = 128
        self.bert_embedding = 768
        self.dropout = 0.5
        self.rnn_layer = 1
        self.lr = 0.0001
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        # self.checkpoint = 'best/RoBERTa_best.pth.tar'
        self.checkpoint = None
        self.epochs = 20
        self.max_grad_norm = 10
        self.patience = 10
        self.seed = 42

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


config = Config()
