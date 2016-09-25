require "spec"
require "csv"
require "../lib/ml"
require "../lib/trees"
require "../lib/array"
require "../lib/random"
require "../lib/math"

def golf_features
  f0 = %i(r r o s s s o r r s r o o s)  # outlook
  f1 = %i(h h h m c c c m c m m m h m)  # temperature
  f2 = %i(h h h h n n n h n n n h n h)  # humidity
  f3 = %i(f t f f f t t f f f t t f t)  # windy
  [f0, f1, f2, f3].transpose
end

describe ML::Classifiers::DecisionTreeClassifier do
  describe "with categorical data" do
    it "should classify" do
      # http://www.saedsayad.com/decision_tree.htm
      x = golf_features
      y =  %i(n n y y y n y n y y y y y n)  # play golf?

      column_names = %i(outlook temperature hummidity windy play_golf)

      trained_tree = ML::Classifiers::DecisionTreeClassifier(Symbol, Symbol).new.fit(x, y)
      # trained_tree.show_tree(column_names: column_names)

      trained_tree.predict([%i(s h h t)]).should eq([:n])
      trained_tree.predict([%i(s h h t)]).should eq([:n])
    end

    it "should classify unseen paths" do
      # http://www.saedsayad.com/decision_tree.htm
      x = golf_features
      y =  %i(n n y y y n y n y y y y y n)  # play golf?

      column_names = %i(outlook temperature hummidity windy play_golf)

      trained_tree = ML::Classifiers::DecisionTreeClassifier(Symbol, Symbol).new.fit(x, y)
      # trained_tree.show_tree(column_names: column_names)

      trained_tree.predict([%i(s s s s)]).should eq([:n])
    end
  end

  describe "with numerical data (iris dataset)" do
    it "should classify" do
      x, y = ML.load_floats_csv("iris.csv")
      clf = ML::Classifiers::DecisionTreeClassifier(Float32, String).new.fit(x, y)
      y_pred = clf.predict(x)

      acc = ML.accuracy(y, y_pred)
      acc.is_a?(Float).should eq(true)
    end
  end

end


describe ML::Classifiers::DecisionTreeRegressor do
  describe "with categorical data (hair eye color)" do
    it "with training data" do
      x, y = ML.load_string_csv("HairEyeColor.csv", columns_to_skip: 1)
      clf = ML::Classifiers::DecisionTreeRegressor(String, Float32).new.fit(x, y)
      y_pred = clf.predict(x)
      y_pred.should eq([32, 53, 10, 3.5, 11, 50, 10, 30, 10, 25, 6.75, 6.75, 2.5, 14.5, 6.75, 6.75, 36, 66, 16, 3.5, 9, 34, 7, 64, 5, 29, 6.75, 6.75, 2.5, 14.5, 6.75, 6.75])
    end
  end

  it "with unseen data" do
    x, y = ML.load_string_csv("HairEyeColor.csv", columns_to_skip: 1)
    train_index, test_index = ML.train_test_split(x.size, train_size: 0.8)

    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    clf = ML::Classifiers::DecisionTreeRegressor(String, Float32).new.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = ML.accuracy(y_test, y_pred)
    acc.is_a?(Float).should eq(true)
  end

  it "using max depth in the trees" do
    x = Random.sequence(80).map {|x| x * 5}
    x.sort!
    y = Math.sin(x)

    seq = Random.sequence(16).map{|x| 3 * (0.5 - x)}
    y.each_with_index do |e, i|
      y[i] += seq[i/5] if i%5 == 0
    end

    regr = ML::Classifiers::DecisionTreeRegressor(Float64, Float64).new(max_depth: 2)

    x = x.map {|xi| [xi]}

    regr.fit(x, y)

    regr.depth.should eq(2)
  end

end
