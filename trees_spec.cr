require "spec"
require "csv"
require "./ml"
require "./trees"
require "./array"

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

      trained_tree = ML::Classifiers::DecisionTreeClassifier.new.fit(x, y)
      trained_tree.class.should eq(ML::Classifiers::DecisionTreeClassifier)
      # trained_tree.show_tree(column_names: column_names)

      trained_tree.predict([%i(s h h t)]).should eq([:n])
      trained_tree.predict([%i(s h h t)]).should eq([:n])
    end

    it "should classify unseen paths" do
      # http://www.saedsayad.com/decision_tree.htm
      x = golf_features
      y =  %i(n n y y y n y n y y y y y n)  # play golf?

      column_names = %w(outlook temperature hummidity windy play_golf)

      trained_tree = ML::Classifiers::DecisionTreeClassifier.new.fit(x, y)
      trained_tree.class.should eq(ML::Classifiers::DecisionTreeClassifier)
      # trained_tree.show_tree(column_names: column_names)

      trained_tree.predict([%w(s s s s)]).should eq(["n"])
    end
    #
    # it "should predict regression" do
    #   # http://www.saedsayad.com/decision_tree_reg.htm
    #   x = golf_features
    #   y =  [25, 30, 46, 45, 52, 23, 43, 35, 38, 46, 48, 52, 44, 30].map(&.to_f32)  # hours played
    #
    #   column_names = %w(outlook temperature hummidity windy hours_played)
    #
    #   trained_tree = ML::Classifiers::DecisionTreeRegresor.new.fit(x, y)
    #
    #   trained_tree.class.should eq(ML::Classifiers::DecisionTreeRegresor)
    #   trained_tree.show_tree(column_names: column_names)
    #   trained_tree.predict([%w(s m n f)]).should eq(47.7)
    # end


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
