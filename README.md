Ejemplo:

```crystal
require "csv"
require "./knn"
require "./array"
require "./ml"

x, y = ML.load_csv("iris.csv")

train_index, test_index = ML.train_test_split(x.size, test_size: 0.2)

x_train, y_train = x[train_index], y[train_index]
x_test, y_test = x[test_index], y[test_index]

puts "Set sizes: x_train #{x_train.size} y_train #{y_train.size} x_test #{x_test.size} y_test #{y_test.size}"



(5..150).step(10).each do |n|
  clf = KNeighborsClassifier(typeof(x.first), typeof(y.first)).new(n_neighbors: n)
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  acc = ML.accuracy(y_test, y_pred)

  p "Accuracy (KNN - #{n} neighbors): #{acc}"
end

```

```
Set sizes: x_train 120 y_train 120 x_test 30 y_test 30
"Accuracy (KNN - 5 neighbors): 1.0"
"Accuracy (KNN - 15 neighbors): 0.96666666666666667"
"Accuracy (KNN - 25 neighbors): 0.96666666666666667"
"Accuracy (KNN - 35 neighbors): 0.96666666666666667"
"Accuracy (KNN - 45 neighbors): 1.0"
"Accuracy (KNN - 55 neighbors): 0.93333333333333335"
"Accuracy (KNN - 65 neighbors): 0.9"
"Accuracy (KNN - 75 neighbors): 0.9"
"Accuracy (KNN - 85 neighbors): 0.8"
"Accuracy (KNN - 95 neighbors): 0.73333333333333328"
"Accuracy (KNN - 105 neighbors): 0.73333333333333328"
"Accuracy (KNN - 115 neighbors): 0.66666666666666663"
"Accuracy (KNN - 125 neighbors): 0.3"
"Accuracy (KNN - 135 neighbors): 0.3"
"Accuracy (KNN - 145 neighbors): 0.3"
```
