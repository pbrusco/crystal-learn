require "spec"
require "csv"
require "../lib/ml"
require "../lib/trees"

# examples comming from: http://www.saedsayad.com/decision_tree.htm
describe ML do
  describe "information on splitting" do
    it "should calculate entropy and gain given categorical features" do
      f1 = %i(r r o s s s o r r s r o o s) # outlook
      f2 = %i(h h h m c c c m c m m m h m) # temperature
      f3 = %i(h h h h n n n h n n n h n h) # humidity
      f4 = %i(f t f f f t t f f f t t f t) # windy
      y = %i(n n y y y n y n y y y y y n)  # play golf?

      ML.entropy(y).should be_close(0.94, delta: 0.01)

      ML.entropy(y, given: f1).should be_close(0.693, delta: 0.01)

      ML.gain(y, given: f1).should be_close(0.247, delta: 0.01)
      ML.gain(y, given: f2).should be_close(0.029, delta: 0.01)
      ML.gain(y, given: f3).should be_close(0.152, delta: 0.01)
      ML.gain(y, given: f4).should be_close(0.048, delta: 0.01)
    end

    it "should calculate std given categorical features and continuous output" do
      f1 = %i(r r o s s s o r r s r o o s)                         # outlook
      f2 = %i(h h h m c c c m c m m m h m)                         # temperature
      f3 = %i(h h h h n n n h n n n h n h)                         # humidity
      f4 = %i(f t f f f t t f f f t t f t)                         # windy
      y = [25, 30, 46, 45, 52, 23, 43, 35, 38, 46, 48, 52, 44, 30] # hours played

      y.std.should be_close(9.32, delta: 0.01)
      ML.std(y, given: f1).should be_close(7.66, delta: 0.01)
      ML.std_reduction(y, given: f1).should be_close(1.66, delta: 0.01)
    end
  end

  describe "normalizing dataset" do
    it "can normalize dataset" do
      x, y = ML.load_floats_csv("iris.csv")
      x_standardized = ML.standardize(x)

      x_standardized.shape.should eq(x.shape)
    end
  end

  describe "kfold_cross_validation" do
    it "can iterate over folds" do
      x = [[1, 2], [3, 4], [1, 2], [3, 4]]
      y = [0, 0, 1, 1]
      skf = ML.kfold_cross_validation(dataset_size: y.size, n_folds: 2, shuffle: false)
      skf.size.should eq(2)

      train_index, test_index = skf.first
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
      train_index.should eq([0, 1])
      test_index.should eq([2, 3])

      train_index, test_index = skf.last
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]

      train_index.should eq([2, 3])
      test_index.should eq([0, 1])
    end
  end
end
