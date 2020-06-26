# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'test_batch_size': 128,# Test Batch size.
    'num_epochs': 100,# Number of epochs to train for.
    'latent_size':100, # latent size of
    'learning_rate': 1e-3,# Learning rate.
    'print_interval' : 100,
    'beta1': 0.5,
    'beta2': 0.999,
    'use_cuda' : True,
    'save_epoch' : 25,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'CelebA', # Dataset to use. Choose from {MNIST, CelebA, CASIA, FERET}. CASE MUST MATCH EXACTLY!!!!!
    'log_path' : 'logs/',
    'model_path' : 'checkpoints/',
    'compare_path' : 'comparisons/'
    }