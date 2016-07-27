require "spec"
require "csv"
require "./ml"
require "./trees"
require "./array"

describe ML::Classifiers::DecisionTreeClassifier do
  describe "can create and predict in simple tree" do
    it "using gain" do
      # http://www.saedsayad.com/decision_tree.htm
      f1 = %w(r r o s s s o r r s r o o s)  # outlook
      f2 = %w(h h h m c c c m c m m m h m)  # temperature
      f3 = %w(h h h h n n n h n n n h n h)  # humidity
      f4 = %w(f t f f f t t f f f t t f t)  # windy
      y =  %w(n n y y y n y n y y y y y n)  # play golf?
      x = [f1, f2, f3, f4].transpose

      column_names = %w(outlook temperature hummidity windy play_golf)

      trained_tree = ML::Classifiers::DecisionTreeClassifier.new.fit(x, y, column_names: column_names)
      trained_tree.class.should eq(ML::Classifiers::DecisionTreeClassifier)
      # trained_tree.show_tree

      trained_tree.predict([%w(s h h t)]).should eq(["n"])
      trained_tree.predict([%w(s h h t)]).should eq(["n"])
    end

    it "can be use for regressions on categorical data (hair eye color)" do
      x, y = ML.load_string_csv("HairEyeColor.csv")

      clf = ML::Classifiers::DecisionTreeRegresor.new
      clf.fit(x, y)
      y_pred = clf.predict(x)
      puts y_pred
    end
    #
    # it "can be use for regressions for continuous data (iris dataset)" do
    #   x, y = ML.load_csv("iris.csv")
    #
    #   train_index, test_index = ML.train_test_split(x.size, train_size: 0.8)
    #
    #   x_train, y_train = x[train_index], y[train_index]
    #   x_test, y_test = x[test_index], y[test_index]
    #
    #   puts "Set sizes: x_train #{x_train.size} y_train #{y_train.size} x_test #{x_test.size} y_test #{y_test.size}"
    #
    #   clf = ML::Classifiers::DecisionTreeClassifier.new
    #   clf.fit(x_train, y_train)
    #   y_pred = clf.predict(x_test)
    #   acc = ML.accuracy(y_test, y_pred)
    #   acc.should eq(0.5)
    # end
  end

end
