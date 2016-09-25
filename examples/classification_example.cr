require "csv"
require "../lib/knn"
require "../lib/trees"
require "../lib/array"
require "../lib/ml"

puts "Loading IRIS dataset"

x, y = ML.load_floats_csv("iris.csv")
puts "Shapes: X: #{x.shape}, y: #{y.shape}"

def folds_accuracy(clf, x, y, *, n_folds k)
  accuracies = [] of Float64
  folds = ML.kfold_cross_validation(n_folds: k, dataset_size: y.size)

  folds.each do |train_index, test_index|
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracies << ML.accuracy(y_test, y_pred)
  end
  accuracies.mean
end

puts "------------------ KNN -------------------"

(5..150).step(10).each do |n|
  clf = ML::Classifiers::KNeighborsClassifier(typeof(x.first.first), typeof(y.first)).new(n_neighbors: n)
  folds_acc = folds_accuracy(clf, x, y, n_folds: 10).round(2)
  puts "10-folds accuracy #{folds_acc} (KNN - #{n} neighbors)"
end

puts "------------------ TREES -------------------"

(2..15).each do |max_depth|
  clf = ML::Classifiers::DecisionTreeClassifier(typeof(x.first.first), typeof(y.first)).new(max_depth: max_depth)
  folds_acc = folds_accuracy(clf, x, y, n_folds: 10).round(2)
  puts "10-folds accuracy #{folds_acc} (DecisionTreeClassifier - max_depth: #{max_depth})"
end
  # uncomment to vizualize the tree:
  # clf.show_tree(%w(sepal_length sepal_width petal_length petal_width species))
puts
