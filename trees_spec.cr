require "spec"
require "./ml"
require "./trees"

describe ML::Classifiers::DecisionTreeClassifier do
  describe "can create simple tree" do
    it "using gain" do
      # http://www.saedsayad.com/decision_tree.htm
      f1 = %i(r r o s s s o r r s r o o s)  # outlook
      f2 = %i(h h h m c c c m c m m m h m)  # temperature
      f3 = %i(h h h h n n n h n n n h n h)  # humidity
      f4 = %i(f t f f f t t f f f t t f t)  # windy
      y =  %i(n n y y y n y n y y y y y n)  # play golf?
      x = [f1, f2, f3, f4].transpose

      column_names = %i(outlook temperature hummidity windy play_golf)

      trained_tree = ML::Classifiers::DecisionTreeClassifier.new.fit(x, y, column_names: column_names)
      trained_tree.class.should eq(ML::Classifiers::DecisionTreeClassifier)
      # trained_tree.show_tree

      trained_tree.predict([%i(s h h t)]).should eq([:n])
      trained_tree.predict([%i(s h h t)]).should eq([:n])
    end
  end

end
