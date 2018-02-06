import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10)):
    '''
    input: 
    x: array-like, shape (n_samples, n_features), training vector
    y: array_like, shape (n_samples, n_features), target relative to X for classification or regression. None for unsupervised learning
    ylim: tuple, hsaple(ymin, ymax), optional, define minimum and maximum yvalues plotted.
    cv: int, cross validation generator or an iterable, optional, determines the cross-validation splitting strategy.
        possible input for cv are:
        - None, to use the default 3-fold cross-validation
        - integer, to specify trhe number of folds.
        - an object to be used as a cross-validation generator.
        - an iterable yielding train/test splits
        for integer/None inputs, if 'y' is binary or multiclass, class 'StratifiedKFold' used.
        if the estimator is not a classifier or if 'y' is neither binary nor multiclass, class 'KFold' is used.
    scoring: string, callable or None, default None
    i.e. log loss is specified as 'neg_log_loss'
    '''
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.xlabel('Training examples')
    plt.ylabel('score')
    plt.ylim((0.6, 1.01))
    #plt.gca().invert_yaxis()
    
    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Test score')
    
    #plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='b')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, alpha=0.1, color='r')
    
    #draw the plot and reset the y-axis
    plt.show()

# Compute confusion matrix
if __name__=='main':
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix=np.array([[ 1.,    0.,    0.  ],[ 0.,    0.62,  0.38], [ 0.,    0.,    1.  ]])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()
    
    plot_learning_curve(clf2, xtrain1, ytrain1)