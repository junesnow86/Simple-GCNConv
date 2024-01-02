from model import RNNModel
from trainer import Trainer

if __name__ == '__main__':

    depths = [3, 5, 7]
    dropout_ps = [0, 0.2, 0.5]
    normalization_modes = ['bn', 'ln', 'in']
    lrd_modes = ['plateau', 'step', 'exponential']
    residual_connections = [False, True]

    tuning_max_acc = 0.0
    # best_depth = -1
    # print(f'>>> Start tuning depth')
    # for depth in depths:
    #     print(f'hyperparameters: depth={depth}, dropout_p={dropout_ps[1]}, normalization_mode={normalization_modes[0]}, lrd_mode={lrd_modes[0]}, residual_connection={residual_connections[1]}')
    #     model = RNNModel(
    #         input_dim=128,
    #         hidden_dim=128,
    #         num_class=5,
    #         depth=depth,
    #     )

    #     trainer = Trainer(model, fig_name='figures/first_try', epochs=20, batch_size=512)

    #     max_val_acc = trainer.train(lr=0.001, num_workers=10, wait=5, lrd=True, lrd_mode='plateau')
    #     print(f'>>> Max val acc: {max_val_acc}')
    #     test_acc = trainer.test()
    #     print(f'>>> Test acc: {test_acc}')

    #     if max_val_acc > tuning_max_acc:
    #         tuning_max_acc = max_val_acc
    #         best_depth = depth
    # print(f'>>> Tuning finished. Best depth: {best_depth}')

    # best_dropout_p = 0
    # print(f'>>> Start tuning dropout_p')
    # for dropout_p in dropout_ps:
    #     print(f'hyperparameters: depth={depths[0]}, dropout_p={dropout_p}, normalization_mode={normalization_modes[0]}, lrd_mode={lrd_modes[0]}, residual_connection={residual_connections[1]}')
    #     model = RNNModel(
    #         input_dim=128,
    #         hidden_dim=128,
    #         num_class=5,
    #         depth=depths[0],
    #         dropout_p=dropout_p,
    #         normalize_mode=normalization_modes[0],
    #         res=residual_connections[1]
    #     )

    #     trainer = Trainer(model, fig_name='figures/first_try', epochs=20, batch_size=512)

    #     max_val_acc = trainer.train(lr=0.001, num_workers=10, wait=5, lrd=True, lrd_mode=lrd_modes[0])
    #     print(f'>>> Max val acc: {max_val_acc}')
    #     test_acc = trainer.test()
    #     print(f'>>> Test acc: {test_acc}')

    #     if max_val_acc > tuning_max_acc:
    #         tuning_max_acc = max_val_acc
    #         best_dropout_p = dropout_p
    # print(f'>>> Tuning finished. Best dropout_p: {best_dropout_p}')

    # best_normalization_mode = 'bn'
    # print(f'>>> Start tuning normalization mode')
    # for normalization_mode in normalization_modes:
    #     print(f'hyperparameters: depth={depths[0]}, dropout_p={dropout_ps[1]}, normalization_mode={normalization_mode}, lrd_mode={lrd_modes[0]}, residual_connection={residual_connections[1]}')
    #     model = RNNModel(
    #         input_dim=128,
    #         hidden_dim=128,
    #         num_class=5,
    #         depth=depths[0],
    #         dropout_p=dropout_ps[1],
    #         normalize_mode=normalization_mode,
    #         res=residual_connections[1]
    #     )

    #     trainer = Trainer(model, fig_name='figures/first_try', epochs=20, batch_size=512)

    #     max_val_acc = trainer.train(lr=0.001, num_workers=10, wait=5, lrd=True, lrd_mode=lrd_modes[0])
    #     print(f'>>> Max val acc: {max_val_acc}')
    #     test_acc = trainer.test()
    #     print(f'>>> Test acc: {test_acc}')

    #     if max_val_acc > tuning_max_acc:
    #         tuning_max_acc = max_val_acc
    #         best_normalization_mode = normalization_mode
    # print(f'>>> Tuning finished. Best normalization mode: {best_normalization_mode}')

    # best_lrd_mode = 'plateau'
    # print(f'>>> Start tuning lrd mode')
    # for lrd_mode in lrd_modes:
    #     print(f'hyperparameters: depth={depths[0]}, dropout_p={dropout_ps[1]}, normalization_mode={normalization_modes[0]}, lrd_mode={lrd_mode}, residual_connection={residual_connections[1]}')
    #     model = RNNModel(
    #         input_dim=128,
    #         hidden_dim=128,
    #         num_class=5,
    #         depth=depths[0],
    #         dropout_p=dropout_ps[1],
    #         normalize_mode=normalization_modes[0],
    #         res=residual_connections[1]
    #     )

    #     trainer = Trainer(model, fig_name='figures/first_try', epochs=20, batch_size=512)

    #     max_val_acc = trainer.train(lr=0.001, num_workers=10, wait=5, lrd=True, lrd_mode=lrd_mode)
    #     print(f'>>> Max val acc: {max_val_acc}')
    #     test_acc = trainer.test()
    #     print(f'>>> Test acc: {test_acc}')

    #     if max_val_acc > tuning_max_acc:
    #         tuning_max_acc = max_val_acc
    #         best_lrd_mode = lrd_mode
    # print(f'>>> Tuning finished. Best lrd mode: {best_lrd_mode}')

    best_res = False
    print(f'>>> Start tuning residual connection')
    for res in residual_connections:
        print(f'hyperparameters: depth={depths[0]}, dropout_p={dropout_ps[1]}, normalization_mode={normalization_modes[0]}, lrd_mode={lrd_modes[0]}, residual_connection={res}')
        model = RNNModel(
            input_dim=128,
            hidden_dim=128,
            num_class=5,
            depth=depths[0],
            dropout_p=dropout_ps[1],
            normalize_mode=normalization_modes[0],
            res=res
        )

        trainer = Trainer(model, fig_name='figures/first_try', epochs=20, batch_size=512)

        max_val_acc = trainer.train(lr=0.001, num_workers=10, wait=5, lrd=True, lrd_mode=lrd_modes[0])
        print(f'>>> Max val acc: {max_val_acc}')
        test_acc = trainer.test()
        print(f'>>> Test acc: {test_acc}')

        if max_val_acc > tuning_max_acc:
            tuning_max_acc = max_val_acc
            best_res = res
    print(f'>>> Tuning finished. Best residual connection: {best_res}')
