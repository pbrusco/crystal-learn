require "csv"
require "./knn"
require "./array"
require "./ml"

f = File.open("iris.csv")

X = [] of Array(Float32)
y = [] of String

CSV.parse(f, separator: ',').each_with_index do |row, idx|
  next if idx == 0
  X << row[0, 4].map {|x| x.to_f32}
  y << row[4]
end

train_index, test_index = ML.train_test_split(X.size, percentage=0.9)

x_train, y_train = X[train_index], y[train_index]
x_test, y_test = X[test_index], y[test_index]

puts "Set sizes: x_train #{x_train.size} y_train #{y_train.size} x_test #{x_test.size} y_test #{y_test.size}"

(5..150).step(10).each do |n|
  clf = KNeighborsClassifier(Float32, String).new(n_neighbors=n)
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  acc = ML.accuracy(y_test, y_pred)

  p "Accuracy (KNN - #{n} neighbors): #{acc}"
end
