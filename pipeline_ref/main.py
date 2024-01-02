from model import RNNModel
from trainer import Trainer

if __name__ == '__main__':
    depth = 7
    dropout_p = 0
    lrd_mode = 'step'
    normalization_mode = 'ln'
    res = True

    model = RNNModel(
        input_dim=128,
        hidden_dim=128,
        num_class=5,
        depth=depth,
        dropout_p=dropout_p,
        normalize_mode=normalization_mode,
        res=res
    )

    trainer = Trainer(model, fig_name='figures/final', epochs=30, batch_size=512)

    max_val_acc = trainer.train(lr=0.001, num_workers=10, wait=5, lrd=True, lrd_mode=lrd_mode)
    print(f'>>> Max val acc: {max_val_acc}')
    test_acc = trainer.test()
    print(f'>>> Test acc: {test_acc}')
