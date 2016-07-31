## An sklearn-like machine-learning library for Crystal

```crystal
require "csv"
require "./knn"
require "./trees"
require "./array"
require "./ml"

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
  clf = ML::Classifiers::KNeighborsClassifier(typeof(x.first), typeof(y.first)).new(n_neighbors: n)
  folds_acc = folds_accuracy(clf, x, y, n_folds: 10).round(2)
  puts "10-folds accuracy #{folds_acc} (KNN - #{n} neighbors)"
end

puts "------------------ TREES -------------------"

(2..15).each do |max_depth|
  clf = ML::Classifiers::DecisionTreeClassifier.new(max_depth: max_depth)
  folds_acc = folds_accuracy(clf, x, y, n_folds: 10).round(2)
  puts "10-folds accuracy #{folds_acc} (DecisionTreeClassifier - max_depth: #{max_depth})"
end
  # uncomment to vizualize the tree:
  # clf.show_tree(%w(sepal_length sepal_width petal_length petal_width species))
puts

```

### Output:
```
Loading IRIS dataset
Shapes: X: {150, 4}, y: 150
------------------ KNN -------------------
10-folds accuracy 0.97 (KNN - 5 neighbors)
10-folds accuracy 0.96 (KNN - 15 neighbors)
10-folds accuracy 0.95 (KNN - 25 neighbors)
10-folds accuracy 0.93 (KNN - 35 neighbors)
10-folds accuracy 0.95 (KNN - 45 neighbors)
10-folds accuracy 0.94 (KNN - 55 neighbors)
10-folds accuracy 0.91 (KNN - 65 neighbors)
10-folds accuracy 0.88 (KNN - 75 neighbors)
10-folds accuracy 0.81 (KNN - 85 neighbors)
10-folds accuracy 0.62 (KNN - 95 neighbors)
10-folds accuracy 0.43 (KNN - 105 neighbors)
10-folds accuracy 0.43 (KNN - 115 neighbors)
10-folds accuracy 0.56 (KNN - 125 neighbors)
10-folds accuracy 0.28 (KNN - 135 neighbors)
10-folds accuracy 0.25 (KNN - 145 neighbors)
------------------ TREES -------------------
10-folds accuracy 0.89 (DecisionTreeClassifier - max_depth: 2)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 3)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 4)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 5)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 6)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 7)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 8)
10-folds accuracy 0.95 (DecisionTreeClassifier - max_depth: 9)
10-folds accuracy 0.94 (DecisionTreeClassifier - max_depth: 10)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 11)
10-folds accuracy 0.93 (DecisionTreeClassifier - max_depth: 12)
10-folds accuracy 0.95 (DecisionTreeClassifier - max_depth: 13)
10-folds accuracy 0.91 (DecisionTreeClassifier - max_depth: 14)
10-folds accuracy 0.95 (DecisionTreeClassifier - max_depth: 15)

```
