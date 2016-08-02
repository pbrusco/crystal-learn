# Based on http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

require "random"
require "../../lib/math"
require "../../lib/random"
require "../../lib/array"
require "../../lib/trees"
require "csv"

x = Random.sequence(80).map {|x| x * 5}
x.sort!
y = Math.sin(x)

seq = Random.sequence(16).map{|x| 3 * (0.5 - x)}
y.each_with_index do |e, i|
  y[i] += seq[i/5] if i%5 == 0
end

regr2 = ML::Classifiers::DecisionTreeRegressor.new(max_depth: 2)
regr5 = ML::Classifiers::DecisionTreeRegressor.new(max_depth: 5)

x = x.map {|xi| [xi]}

regr2.fit(x, y)
regr5.fit(x, y)

# Predict
x_test = ML.arange(0.0, 5.0, step: 0.01).map {|x| [x]}
y_pred2 = regr2.predict(x_test).map {|x| x.round(2)}
y_pred5 = regr5.predict(x_test).map {|x| x.round(2)}

puts "regresor max_depth: 2"
regr2.show_tree(column_names: ["x", "y"])

# puts "regresor max_depth: 5"
# regr5.show_tree(column_names: ["x", "y"])


f = File.open("regressor.csv", mode: "w")

result = CSV.build(f) do |csv|
  x.zip(y).each do |x_i, y_i|
    csv.row x_i[0], y_i, "true"
  end
  x_test.zip(y_pred2).each do |x_i, y_i|
    csv.row x_i[0], y_i, "pred_2"
  end
  x_test.zip(y_pred5).each do |x_i, y_i|
    csv.row x_i[0], y_i, "pred_5"
  end
end

f.close()
