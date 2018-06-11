import matplotlib.pyplot as plt


def show_loss_acc_picture(unit_count, round_of_run, train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    plt.figure()
    title = "Loss 1 layer, %d unit" % unit_count
    plt.plot(range(round_of_run), train_loss_list)
    plt.plot(range(round_of_run), test_loss_list)
    plt.title(title, color='blue', wrap=True)
    plt.legend(['train', 'test'])
    plt.figure()
    title = "Acc 1 layer, %d unit" % unit_count
    plt.plot(range(round_of_run), train_acc_list, label='train')
    plt.plot(range(round_of_run), test_acc_list, label='test')
    plt.title(title, color='blue', wrap=True)
    plt.show()


def show_loss_acc_for_two_model(unit_count, round_of_run,
                                train_loss_1_list, train_loss_2_list,
                                test_loss_1_list, test_loss_2_list,
                                train_acc_1_list, train_acc_2_list,
                                test_acc_1_list, test_acc_2_list,
                                model_name_1, model_name_2):
    plt.figure()
    title = "Loss 1 layer Train, %d unit" % unit_count
    plt.plot(range(round_of_run), train_loss_1_list)
    plt.plot(range(round_of_run), train_loss_2_list)
    plt.title(title, color='blue', wrap=True)
    plt.legend([model_name_1, model_name_2])

    plt.figure()
    title = "Loss 1 layer Test, %d unit" % unit_count
    plt.plot(range(round_of_run), test_loss_1_list)
    plt.plot(range(round_of_run), test_loss_2_list)
    plt.title(title, color='blue', wrap=True)
    plt.legend([model_name_1, model_name_2])

    plt.figure()
    title = "Acc 1 layer Train, %d unit" % unit_count
    plt.plot(range(round_of_run), train_acc_1_list, label=model_name_1)
    plt.plot(range(round_of_run), train_acc_2_list, label=model_name_2)
    plt.title(title, color='blue', wrap=True)
    plt.legend([model_name_1, model_name_2])

    plt.figure()
    title = "Acc 1 layer Test, %d unit" % unit_count
    plt.plot(range(round_of_run), test_acc_1_list, label=model_name_1)
    plt.plot(range(round_of_run), test_acc_2_list, label=model_name_2)
    plt.title(title, color='blue', wrap=True)
    plt.legend([model_name_1, model_name_2])
    plt.show()
