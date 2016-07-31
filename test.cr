require "csv"
require "./knn"
require "./trees"
require "./array"
require "./ml"

x, y = ML.load_floats_csv("iris.csv")

puts x.shape
puts y.shape

def folds_accuracy(clf, x, y, *, n_folds k)
  accuracies = [] of Float64
  folds = ML.kfold_cross_validation(y, n_folds: k)

  folds.each do |train_index, test_index|
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracies << ML.accuracy(y_test, y_pred)
  end
  accuracies
end

(5..150).step(10).each do |n|
  clf = ML::Classifiers::KNeighborsClassifier(typeof(x.first), typeof(y.first)).new(n_neighbors: n)
  folds_acc = folds_accuracy(clf, x, y, folds: 10)
  p "Accuracy (KNN): #{folds_acc} (10 folds) (#{n} neighbors)"
end

clf = ML::Classifiers::DecisionTreeClassifier.new
folds_acc = folds_accuracy(clf, x, y, folds: 10)
p "Accuracy (DecisionTreeClassifier): #{acc} (10 folds)"
# clf.show_tree(%w(sepal_length sepal_width petal_length petal_width species))
