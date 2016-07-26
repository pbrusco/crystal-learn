require "csv"
require "./knn"
require "./array"
require "./ml"

x, y = ML.load_csv("iris.csv")

train_index, test_index = ML.train_test_split(x.size, train_size: 0.8)

x_train, y_train = x[train_index], y[train_index]
x_test, y_test = x[test_index], y[test_index]

puts "Set sizes: x_train #{x_train.size} y_train #{y_train.size} x_test #{x_test.size} y_test #{y_test.size}"

(5..150).step(10).each do |n|
  knn = ML::Classifiers::KNeighborsClassifier(typeof(x.first), typeof(y.first))
  clf = knn.new(n_neighbors: n)
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  acc = ML.accuracy(y_test, y_pred)

  p "Accuracy (KNN - #{n} neighbors): #{acc}"
end
