from tensorboard import program


def tensorboard_launch(experiments_folder):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', experiments_folder])
    url = tb.launch()
    print(f'TensorBoard at {url}')
